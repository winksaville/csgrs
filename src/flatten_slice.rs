use crate::csg::CSG;
use std::fmt::Debug;
use geo::{GeometryCollection, LineString, orient::Direction, Geometry, Polygon as GeoPolygon, Orient, coord, MultiPolygon, BooleanOps};
use hashbrown::HashMap;
use crate::plane::Plane;
use crate::bsp::Node;
use crate::float_types::{Real, EPSILON};
use nalgebra::Point3;
use crate::vertex::Vertex;

impl<S: Clone + Debug> CSG<S> where S: Clone + Send + Sync {
    /// Flattens any 3D polygons by projecting them onto the XY plane (z=0),
    /// unifies them into one or more 2D polygons, and returns a purely 2D CSG.
    ///
    /// - If this CSG is already 2D (`self.polygons` is empty), just returns `self.clone()`.
    /// - Otherwise, all `polygons` are tessellated, projected into XY, and unioned.
    /// - We also union any existing 2D geometry (`self.geometry`).
    /// - The output has `.polygons` empty and `.geometry` containing the final 2D shape.
    pub fn flatten(&self) -> CSG<S> {
        // 1) If there are no 3D polygons, this is already purely 2D => return as-is
        if self.polygons.is_empty() {
            return self.clone();
        }
    
        // 2) Convert all 3D polygons into a collection of 2D polygons
        let mut flattened_3d = Vec::new(); // will store geo::Polygon<Real>
    
        for poly in &self.polygons {
            // Tessellate this polygon into triangles
            let triangles = poly.tessellate();
            // Each triangle has 3 vertices [v0, v1, v2].
            // Project them onto XY => build a 2D polygon (triangle).
            for tri in triangles {
                let ring = vec![
                    (tri[0].pos.x, tri[0].pos.y),
                    (tri[1].pos.x, tri[1].pos.y),
                    (tri[2].pos.x, tri[2].pos.y),
                    (tri[0].pos.x, tri[0].pos.y), // close ring explicitly
                ];
                let polygon_2d = geo::Polygon::new(LineString::from(ring), vec![]);
                flattened_3d.push(polygon_2d);
            }
        }
    
        // 3) Union all these polygons together into one MultiPolygon
        //    (We could chain them in a fold-based union.)
        let unioned_from_3d = if flattened_3d.is_empty() {
            MultiPolygon::new(Vec::new())
        } else {
            // Start with the first polygon as a MultiPolygon
            let mut mp_acc = MultiPolygon(vec![flattened_3d[0].clone()]);
            // Union in the rest
            for p in flattened_3d.iter().skip(1) {
                mp_acc = mp_acc.union(&MultiPolygon(vec![p.clone()]));
            }
            mp_acc
        };
    
        // 4) Union this with any existing 2D geometry (polygons) from self.geometry
        let existing_2d = &self.to_multipolygon();  // turns geometry -> MultiPolygon
        let final_union = unioned_from_3d.union(existing_2d);
        // Optionally ensure consistent orientation (CCW for exteriors):
        let oriented = final_union.orient(Direction::Default);
    
        // 5) Store final polygons as a MultiPolygon in a new GeometryCollection
        let mut new_gc = GeometryCollection::default();
        new_gc.0.push(Geometry::MultiPolygon(oriented));
    
        // 6) Return a purely 2D CSG: polygons empty, geometry has the final shape
        CSG {
            polygons: Vec::new(),
            geometry: new_gc,
            metadata: self.metadata.clone(),
        }
    }
    
    /// Slice this solid by a given `plane`, returning a new `CSG` whose polygons
    /// are either:
    /// - The polygons that lie exactly in the slicing plane (coplanar), or
    /// - Polygons formed by the intersection edges (each a line, possibly open or closed).
    ///
    /// The returned `CSG` can contain:
    /// - **Closed polygons** that are coplanar,
    /// - **Open polygons** (poly-lines) if the plane cuts through edges,
    /// - Potentially **closed loops** if the intersection lines form a cycle.
    ///
    /// # Example
    /// ```
    /// let cylinder = CSG::cylinder(1.0, 2.0, 32, None);
    /// let plane_z0 = Plane { normal: Vector3::z(), w: 0.0 };
    /// let cross_section = cylinder.slice(plane_z0);
    /// // `cross_section` will contain:
    /// //   - Possibly an open or closed polygon(s) at z=0
    /// //   - Or empty if no intersection
    /// ```
    pub fn slice(&self, plane: Plane) -> CSG<S> {
        // Build a BSP from all of our polygons:
        let node = Node::new(&self.polygons.clone());

        // Ask the BSP for coplanar polygons + intersection edges:
        let (coplanar_polys, intersection_edges) = node.slice(&plane);

        // “Knit” those intersection edges into polylines. Each edge is [vA, vB].
        let polylines_3d = unify_intersection_edges(&intersection_edges);

        // Convert each polyline of vertices into a Polygon<S>
        let mut result_polygons = Vec::new();

        // Add the coplanar polygons. We can re‐assign their plane to `plane` to ensure
        // they share the exact plane definition (in case of numeric drift).
        for mut p in coplanar_polys {
            p.plane = plane.clone(); // unify plane data
            result_polygons.push(p);
        }

        let mut new_gc = GeometryCollection::default();

        // Convert the “chains” or loops into open/closed polygons
        for mut chain in polylines_3d {
            let n = chain.len();
            if n < 2 {
                // degenerate
                continue;
            }

            // check if first and last point are within EPSILON of each other
            let dist_sq = (chain[0].pos - chain[n - 1].pos).norm_squared();
            if dist_sq < EPSILON * EPSILON {
                // Force them to be exactly the same, closing the line
                chain[n - 1] = chain[0].clone();
            }
            
            let polyline = LineString::new(chain.iter().map(|vertex| {coord! {x: vertex.pos.x, y: vertex.pos.y}}).collect());
            
            if polyline.is_closed() {
                let polygon = GeoPolygon::new(polyline, vec![]);
                let oriented = polygon.orient(Direction::Default);
                new_gc.0.push(Geometry::Polygon(oriented));
            } else {
                new_gc.0.push(Geometry::LineString(polyline));
            }
        }

        // Return a purely 2D CSG: polygons empty, geometry has the final shape
        CSG {
            polygons: Vec::new(),
            geometry: new_gc,
            metadata: self.metadata.clone(),
        }
    }
    
    /// Checks if the CSG object is manifold.
    ///
    /// This function:
    ///   1) Triangulates the shape.
    ///   2) Gathers each triangle’s 3 edges and counts occurrences of each (using approximate ordering).
    ///   3) Adjusts edges with count==1 if they’re collinear merges of other count==1 edges.
    ///   4) Returns `true` if every edge ends with count==2, else `false`.
    ///
    /// TODOs not fully implemented here:
    ///   - Consistent orientation / no flipped neighbors
    ///   - Zero-area triangle check
    ///
    /// # Returns
    ///
    /// - `true`: If the CSG object is manifold (all edges appear exactly 2 times).
    /// - `false`: If the CSG object is not manifold.
    pub fn is_manifold(&self) -> bool {
        fn approx_lt(a: &Point3<f64>, b: &Point3<f64>) -> bool {
            // Compare x
            if (a.x - b.x).abs() > EPSILON {
                return a.x < b.x;
            }
            // If x is "close", compare y
            if (a.y - b.y).abs() > EPSILON {
                return a.y < b.y;
            }
            // If y is also close, compare z
            a.z < b.z
        }

        // Convert a 3D point into a string with limited decimal places, to be used as a HashMap key
        fn point_key(p: &Point3<f64>) -> String {
            // Truncate/round to e.g. 6 decimals
            format!("{:.6},{:.6},{:.6}", p.x, p.y, p.z)
        }

        // A small helper to restore a sorted (keyA, keyB) from two 3D points
        fn make_edge_key(p0: &Point3<f64>, p1: &Point3<f64>) -> (String, String) {
            if approx_lt(p0, p1) {
                (point_key(p0), point_key(p1))
            } else {
                (point_key(p1), point_key(p0))
            }
        }

        // Check whether three points are collinear, within EPSILON tolerance
        fn collinear(p0: &Point3<f64>, p1: &Point3<f64>, p2: &Point3<f64>) -> bool {
            // A simple way is to check the cross product of (p1 - p0) and (p2 - p0)
            // If its magnitude is close to zero, they're collinear.
            let v0 = p1 - p0;
            let v1 = p2 - p0;
            let cross = v0.cross(&v1);
            cross.norm() < EPSILON
        }

        // Triangulate the whole shape
        let tri_csg = self.tessellate();

        // First gather every directed edge, storing them in a map of (sortedKey) -> count
        let mut edge_counts: HashMap<(String, String), u32> = HashMap::new();

        for poly in &tri_csg.polygons {
            // Each triangulated polygon has 3 vertices: [v0, v1, v2]
            for &(i0, i1) in &[(0, 1), (1, 2), (2, 0)] {
                let p0 = &poly.vertices[i0].pos;
                let p1 = &poly.vertices[i1].pos;

                // Sort so that (p0,p1) == (p1,p0) for hashing
                let key = make_edge_key(p0, p1);

                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }

        //
        // --- NEW STEP: attempt to "patch" edges that appear once by checking if they are collinear
        // with another single-count edge, forming a larger single-count edge. If so, mark all 3 edges
        // as count = 2.
        //

        // Collect all edges that have count == 1 along with their original Point3<f64> data.
        // We’ll store them in a vector of ( (keyA, keyB), (pointA, pointB) ).
        // That way we can easily iterate and find collinear neighbors.
        let mut single_edges = Vec::new();
        // Also keep a small helper map so we can quickly find the original points from a given key:
        let mut key_to_points = HashMap::new();

        // We'll need to re-generate the original points from each edge key. If you have them around
        // you can store them more directly, but for demonstration we re-scan the polygons:
        // (Alternatively, you could store them while building edge_counts in the first place.)

        // Make a temporary structure (string -> Point3) for every vertex in tri_csg,
        // so we can quickly map back from "x.xx,y.yy,z.zz" to a real Point3.
        let mut vertex_map: HashMap<String, Point3<f64>> = HashMap::new();
        for poly in &tri_csg.polygons {
            for v in &poly.vertices {
                let k = point_key(&v.pos);
                // Overwrite duplicates, that's fine
                vertex_map.insert(k, v.pos);
            }
        }

        // Now fill single_edges and key_to_points
        for ((k_a, k_b), &count) in &edge_counts {
            if count == 1 {
                if let (Some(p_a), Some(p_b)) = (vertex_map.get(k_a), vertex_map.get(k_b)) {
                    single_edges.push(((k_a.clone(), k_b.clone()), (*p_a, *p_b)));
                    key_to_points.insert((k_a.clone(), k_b.clone()), (*p_a, *p_b));
                }
            }
        }

        // Attempt to patch: for each pair of edges that share a common endpoint, check collinearity
        // and see if the combined edge is also a single-count edge. If so, set all to count=2.
        // We do this in a brute force manner; we might want a more efficient adjacency-based approach
        for i in 0..single_edges.len() {
            let ((k1a, k1b), (p1a, p1b)) = &single_edges[i];
            // Skip if we already changed the count for that edge in a previous iteration
            if edge_counts.get(&(k1a.clone(), k1b.clone())) != Some(&1) {
                continue;
            }

            for j in (i + 1)..single_edges.len() {
                let ((k2a, k2b), (p2a, p2b)) = &single_edges[j];
                if edge_counts.get(&(k2a.clone(), k2b.clone())) != Some(&1) {
                    continue;
                }

                // We want to see if these two edges share exactly one endpoint, and are collinear
                // Suppose edge1 is (p1a, p1b), edge2 is (p2a, p2b).
                // Identify the shared point, if any.
                // Because ordering is not guaranteed (p1a could match p2b, etc.), we'll check all combos.
                let combo = [
                    (p1a, p1b, p2a, p2b),
                    (p1a, p1b, p2b, p2a),
                ];

                let mut made_patch = false;
                'combo_loop: for (e1a, e1b, e2a, e2b) in combo {
                    // check if e1b == e2a is the shared point
                    if (e1b - e2a).norm() < EPSILON {
                        // They share e1b == e2a
                        // Check collinearity about that point
                        // e1a --- e1b/e2a --- e2b
                        if collinear(e1b, e1a, e2b) {
                            // The combined edge is (e1a, e2b), in sorted form
                            let combined_key = make_edge_key(e1a, e2b);

                            // If that combined edge has count == 1, we patch all of them to count=2
                            if edge_counts.get(&combined_key) == Some(&1) {
                                // Mark them as 2
                                *edge_counts.get_mut(&(k1a.clone(), k1b.clone())).unwrap() = 2;
                                *edge_counts.get_mut(&(k2a.clone(), k2b.clone())).unwrap() = 2;
                                *edge_counts.get_mut(&combined_key).unwrap() = 2;
                                made_patch = true;
                                break 'combo_loop;
                            }
                        }
                    }
                }

                // If we already patched using edges i, j, no need to check more combos
                if made_patch {
                    break;
                }
            }
        }

        //println!("{:#?}", edge_counts);

        // Finally, for a perfectly closed manifold surface (with no boundary),
        // each edge should appear exactly 2 times.
        // Now that we've done the extra check for collinear merges, see if everything is 2:
        //edge_counts.values().all(|&count| count == 2)

        // show edges that non-two counts
        let mut non_two_count = 0;
        let total_edge_count = edge_counts.len();
        for ((p0, p1), count) in edge_counts {
            if count != 2 {
                eprintln!("Edge {:?} -> {:?} appears {} time(s)", p0, p1, count);
                non_two_count += 1;
            }
        }

        if non_two_count != 0 {
            eprintln!("non_two_count: {non_two_count} of {total_edge_count} edges");
            false
        } else {
            true
        }
    }
}

// Build a small helper for hashing endpoints:
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct EndKey(i64, i64, i64);

/// Round a floating to a grid for hashing
fn quantize(x: Real) -> i64 {
    // For example, scale by 1e8
    (x * 1e8).round() as i64
}

/// Convert a Vertex’s position to an EndKey
fn make_key(pos: &Point3<Real>) -> EndKey {
    EndKey(quantize(pos.x), quantize(pos.y), quantize(pos.z))
}

/// Take a list of intersection edges `[Vertex;2]` and merge them into polylines.
/// Each edge is a line segment between two 3D points.  We want to “knit” them together by
/// matching endpoints that lie within EPSILON of each other, forming either open or closed chains.
///
/// This returns a `Vec` of polylines, where each polyline is a `Vec<Vertex>`.
fn unify_intersection_edges(edges: &[[Vertex; 2]]) -> Vec<Vec<Vertex>> {
    // We will store adjacency by a “key” that identifies an endpoint up to EPSILON,
    // then link edges that share the same key.

    // Adjacency map: key -> list of (edge_index, is_start_or_end)
    // We’ll store “(edge_idx, which_end)” as which_end = 0 or 1 for edges[edge_idx][0/1].
    let mut adjacency: HashMap<EndKey, Vec<(usize, usize)>> = HashMap::new();

    // Collect all endpoints
    for (i, edge) in edges.iter().enumerate() {
        for end_idx in 0..2 {
            let v = &edge[end_idx];
            let k = make_key(&v.pos);
            adjacency.entry(k).or_default().push((i, end_idx));
        }
    }

    // We’ll keep track of which edges have been “visited” in the final polylines.
    let mut visited = vec![false; edges.len()];

    let mut chains: Vec<Vec<Vertex>> = Vec::new();

    // For each edge not yet visited, we “walk” outward from one end, building a chain
    for start_edge_idx in 0..edges.len() {
        if visited[start_edge_idx] {
            continue;
        }
        // Mark it visited
        visited[start_edge_idx] = true;

        // Our chain starts with `edges[start_edge_idx]`. We can build a small function to “walk”:
        // We’ll store it in the direction edge[0] -> edge[1]
        let e = &edges[start_edge_idx];
        let mut chain = vec![e[0].clone(), e[1].clone()];

        // We walk “forward” from edge[1] if possible
        extend_chain_forward(&mut chain, &adjacency, &mut visited, edges);

        // We also might walk “backward” from edge[0], but
        // we can do that by reversing the chain at the end if needed. Alternatively,
        // we can do a separate pass.  Let’s do it in place for clarity:
        chain.reverse();
        extend_chain_forward(&mut chain, &adjacency, &mut visited, edges);
        // Then reverse back so it goes in the original direction
        chain.reverse();

        chains.push(chain);
    }

    chains
}

/// Extends a chain “forward” by repeatedly finding any unvisited edge that starts
/// at the chain’s current end vertex.
fn extend_chain_forward(
    chain: &mut Vec<Vertex>,
    adjacency: &HashMap<EndKey, Vec<(usize, usize)>>,
    visited: &mut [bool],
    edges: &[[Vertex; 2]],
) {
    loop {
        // The chain’s current end point:
        let last_v = chain.last().unwrap();
        let key = make_key(&last_v.pos);

        // Find candidate edges that share this endpoint
        let Some(candidates) = adjacency.get(&key) else {
            break;
        };

        // Among these candidates, we want one whose “other endpoint” we can follow
        // and is not visited yet.
        let mut found_next = None;
        for &(edge_idx, end_idx) in candidates {
            if visited[edge_idx] {
                continue;
            }
            // If this is edges[edge_idx][end_idx], the “other” end is edges[edge_idx][1-end_idx].
            // We want that other end to continue the chain.
            let other_end_idx = 1 - end_idx;
            let next_vertex = &edges[edge_idx][other_end_idx];

            // But we must also confirm that the last_v is indeed edges[edge_idx][end_idx]
            // (within EPSILON) which we have checked via the key, so likely yes.

            // Mark visited
            visited[edge_idx] = true;
            found_next = Some(next_vertex.clone());
            break;
        }

        match found_next {
            Some(v) => {
                chain.push(v);
            }
            None => {
                break;
            }
        }
    }
}

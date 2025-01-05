//! csg.rs
//!
//! A Rust translation of Evan Wallace's JavaScript CSG code, modified to use
//! nalgebra for points/vectors. This uses a BSP tree for constructive solid
//! geometry (union, subtract, intersect).
//!
//! # Example
//!
//! ```rust
//! let cube = CSG::cube(None);
//! let sphere = CSG::sphere(None);
//! let result = cube.subtract(&sphere);
//! let polygons = result.to_polygons();
//! println!("{}", result.to_stl("my_solid"));
//! ```

use nalgebra::Vector3;

const EPSILON: f64 = 1e-5;

/// A small helper for linear interpolation of `Vec3`.
fn lerp_vec3(a: &Vector3<f64>, b: &Vector3<f64>, t: f64) -> Vector3<f64> {
    a + (b - a) * t
}

/// A vertex of a polygon, holding position and normal.
#[derive(Debug, Clone)]
pub struct Vertex {
    pub pos: Vector3<f64>,
    pub normal: Vector3<f64>,
}

impl Vertex {
    pub fn new(pos: Vector3<f64>, normal: Vector3<f64>) -> Self {
        Vertex { pos, normal }
    }

    pub fn clone(&self) -> Self {
        Vertex {
            pos: self.pos,
            normal: self.normal,
        }
    }

    /// Flip orientation-specific data (like normals)
    pub fn flip(&mut self) {
        self.normal = -self.normal;
    }

    /// Linearly interpolate between self and `other` by parameter `t`
    pub fn interpolate(&self, other: &Vertex, t: f64) -> Vertex {
        Vertex {
            pos: lerp_vec3(&self.pos, &other.pos, t),
            normal: lerp_vec3(&self.normal, &other.normal, t),
        }
    }
}

/// A plane in 3D space defined by a normal and a w-value
#[derive(Debug, Clone)]
pub struct Plane {
    pub normal: Vector3<f64>,
    pub w: f64,
}

impl Plane {
    /// Create a plane from three points
    pub fn from_points(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> Plane {
        let n = (b - a).cross(&(c - a)).normalize();
        Plane {
            normal: n,
            w: n.dot(a),
        }
    }

    pub fn clone(&self) -> Self {
        Plane {
            normal: self.normal,
            w: self.w,
        }
    }

    pub fn flip(&mut self) {
        self.normal = -self.normal;
        self.w = -self.w;
    }

    /// Split `polygon` by this plane if needed, distributing the results into
    /// `coplanar_front`, `coplanar_back`, `front`, and `back`.
    pub fn split_polygon(
        &self,
        polygon: &Polygon,
        coplanar_front: &mut Vec<Polygon>,
        coplanar_back: &mut Vec<Polygon>,
        front: &mut Vec<Polygon>,
        back: &mut Vec<Polygon>,
    ) {
        const COPLANAR: i32 = 0;
        const FRONT: i32 = 1;
        const BACK: i32 = 2;
        const SPANNING: i32 = 3;

        let mut polygon_type = 0;
        let mut types = Vec::with_capacity(polygon.vertices.len());

        // Classify each vertex
        for v in &polygon.vertices {
            let t = self.normal.dot(&v.pos) - self.w;
            let vertex_type = if t < -EPSILON {
                BACK
            } else if t > EPSILON {
                FRONT
            } else {
                COPLANAR
            };
            polygon_type |= vertex_type;
            types.push(vertex_type);
        }

        match polygon_type {
            COPLANAR => {
                // Coincident normals => belongs in front vs. back
                if self.normal.dot(&polygon.plane.normal) > 0.0 {
                    coplanar_front.push(polygon.clone());
                } else {
                    coplanar_back.push(polygon.clone());
                }
            }
            FRONT => {
                front.push(polygon.clone());
            }
            BACK => {
                back.push(polygon.clone());
            }
            _ => {
                // SPANNING
                let mut f: Vec<Vertex> = Vec::new();
                let mut b: Vec<Vertex> = Vec::new();
                let vcount = polygon.vertices.len();

                for i in 0..vcount {
                    let j = (i + 1) % vcount;
                    let ti = types[i];
                    let tj = types[j];
                    let vi = &polygon.vertices[i];
                    let vj = &polygon.vertices[j];

                    if ti != BACK {
                        f.push(vi.clone());
                    }
                    if ti != FRONT {
                        b.push(vi.clone());
                    }

                    if (ti | tj) == SPANNING {
                        let denom = self.normal.dot(&(vj.pos - vi.pos));
                        // Avoid dividing by zero
                        if denom.abs() > EPSILON {
                            let t = (self.w - self.normal.dot(&vi.pos)) / denom;
                            let v = vi.interpolate(vj, t);
                            f.push(v.clone());
                            b.push(v);
                        }
                    }
                }

                if f.len() >= 3 {
                    front.push(Polygon::new(f, polygon.shared.clone()));
                }
                if b.len() >= 3 {
                    back.push(Polygon::new(b, polygon.shared.clone()));
                }
            }
        }
    }
}

/// A convex polygon, defined by a list of vertices and a plane
#[derive(Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Vertex>,
    /// This can hold any “shared” data (color, surface ID, etc.).
    pub shared: Option<String>,
    pub plane: Plane,
}

impl Polygon {
    /// Create a polygon from vertices
    pub fn new(vertices: Vec<Vertex>, shared: Option<String>) -> Self {
        let plane = Plane::from_points(
            &vertices[0].pos,
            &vertices[1].pos,
            &vertices[2].pos,
        );
        Polygon {
            vertices,
            shared,
            plane,
        }
    }

    pub fn clone(&self) -> Self {
        Polygon {
            vertices: self.vertices.iter().map(Vertex::clone).collect(),
            shared: self.shared.clone(),
            plane: self.plane.clone(),
        }
    }

    pub fn flip(&mut self) {
        self.vertices.reverse();
        for v in &mut self.vertices {
            v.flip();
        }
        self.plane.flip();
    }

    /// Triangulate this polygon into a list of triangles, each triangle is [v0, v1, v2].
    pub fn triangulate(&self) -> Vec<[Vertex; 3]> {
        let mut triangles = Vec::new();
        if self.vertices.len() < 3 {
            return triangles;
        }
        let v0 = self.vertices[0].clone();
        for i in 1..(self.vertices.len() - 1) {
            triangles.push([
                v0.clone(),
                self.vertices[i].clone(),
                self.vertices[i + 1].clone(),
            ]);
        }
        triangles
    }
}

/// A BSP tree node, containing polygons plus optional front/back subtrees
#[derive(Debug, Clone)]
pub struct Node {
    pub plane: Option<Plane>,
    pub front: Option<Box<Node>>,
    pub back: Option<Box<Node>>,
    pub polygons: Vec<Polygon>,
}

impl Node {
    pub fn new(polygons: Vec<Polygon>) -> Self {
        let mut node = Node {
            plane: None,
            front: None,
            back: None,
            polygons: Vec::new(),
        };
        if !polygons.is_empty() {
            node.build(&polygons);
        }
        node
    }

    pub fn clone(&self) -> Self {
        Node {
            plane: self.plane.clone(),
            front: self.front.clone(),
            back: self.back.clone(),
            polygons: self.polygons.iter().map(|p| p.clone()).collect(),
        }
    }

    /// Invert all polygons in the BSP tree
    pub fn invert(&mut self) {
        for p in &mut self.polygons {
            p.flip();
        }
        if let Some(ref mut plane) = self.plane {
            plane.flip();
        }
        if let Some(ref mut front) = self.front {
            front.invert();
        }
        if let Some(ref mut back) = self.back {
            back.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    /// Recursively remove all polygons in `polygons` that are inside this BSP tree
    pub fn clip_polygons(&self, polygons: &[Polygon]) -> Vec<Polygon> {
        if self.plane.is_none() {
            return polygons.to_vec();
        }

        let plane = self.plane.as_ref().unwrap();
        let mut front: Vec<Polygon> = Vec::new();
        let mut back: Vec<Polygon> = Vec::new();

        for poly in polygons {
            plane.split_polygon(
                poly,
                &mut Vec::new(), // coplanar_front
                &mut Vec::new(), // coplanar_back
                &mut front,
                &mut back,
            );
        }

        if let Some(ref f) = self.front {
            front = f.clip_polygons(&front);
        }
        if let Some(ref b) = self.back {
            back = b.clip_polygons(&back);
        } else {
            back.clear();
        }

        front.extend(back);
        front
    }

    /// Remove all polygons in this BSP tree that are inside the other BSP tree
    pub fn clip_to(&mut self, bsp: &Node) {
        self.polygons = bsp.clip_polygons(&self.polygons);
        if let Some(ref mut front) = self.front {
            front.clip_to(bsp);
        }
        if let Some(ref mut back) = self.back {
            back.clip_to(bsp);
        }
    }

    /// Return all polygons in this BSP tree
    pub fn all_polygons(&self) -> Vec<Polygon> {
        let mut result = self.polygons.clone();
        if let Some(ref front) = self.front {
            result.extend(front.all_polygons());
        }
        if let Some(ref back) = self.back {
            result.extend(back.all_polygons());
        }
        result
    }

    /// Build a BSP tree from the given polygons
    pub fn build(&mut self, polygons: &[Polygon]) {
        if polygons.is_empty() {
            return;
        }

        if self.plane.is_none() {
            self.plane = Some(polygons[0].plane.clone());
        }
        let plane = self.plane.clone().unwrap();

        let mut front: Vec<Polygon> = Vec::new();
        let mut back: Vec<Polygon> = Vec::new();

        for p in polygons {
            let mut coplanar_front = Vec::new();
            let mut coplanar_back = Vec::new();

            plane.split_polygon(
                p,
                &mut coplanar_front,
                &mut coplanar_back,
                &mut front,
                &mut back,
            );

            self.polygons.append(&mut coplanar_front);
            self.polygons.append(&mut coplanar_back);
        }

        if !front.is_empty() {
            if self.front.is_none() {
                self.front = Some(Box::new(Node::new(vec![])));
            }
            self.front.as_mut().unwrap().build(&front);
        }

        if !back.is_empty() {
            if self.back.is_none() {
                self.back = Some(Box::new(Node::new(vec![])));
            }
            self.back.as_mut().unwrap().build(&back);
        }
    }
}

/// The main CSG solid structure. Contains a list of polygons.
#[derive(Debug, Clone)]
pub struct CSG {
    pub polygons: Vec<Polygon>,
}

impl CSG {
    /// Create an empty CSG
    pub fn new() -> Self {
        CSG {
            polygons: Vec::new(),
        }
    }

    /// Build a CSG from an existing polygon list
    pub fn from_polygons(polygons: Vec<Polygon>) -> Self {
        let mut csg = CSG::new();
        csg.polygons = polygons;
        csg
    }

    /// Clone this CSG
    pub fn clone(&self) -> Self {
        CSG {
            polygons: self.polygons.iter().map(|p| p.clone()).collect(),
        }
    }

    /// Return the internal polygons
    pub fn to_polygons(&self) -> &[Polygon] {
        &self.polygons
    }

    /// CSG union: this ∪ other
    pub fn union(&self, other: &CSG) -> CSG {
        let mut a = Node::new(self.clone().polygons);
        let mut b = Node::new(other.clone().polygons);

        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(&b.all_polygons());

        CSG::from_polygons(a.all_polygons())
    }

    /// CSG subtract: this \ other
    pub fn subtract(&self, other: &CSG) -> CSG {
        let mut a = Node::new(self.clone().polygons);
        let mut b = Node::new(other.clone().polygons);

        a.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        b.invert();
        b.clip_to(&a);
        b.invert();
        a.build(&b.all_polygons());
        a.invert();

        CSG::from_polygons(a.all_polygons())
    }

    /// CSG intersect: this ∩ other
    pub fn intersect(&self, other: &CSG) -> CSG {
        let mut a = Node::new(self.clone().polygons);
        let mut b = Node::new(other.clone().polygons);

        a.invert();
        b.clip_to(&a);
        b.invert();
        a.clip_to(&b);
        b.clip_to(&a);
        a.build(&b.all_polygons());
        a.invert();

        CSG::from_polygons(a.all_polygons())
    }

    /// Invert this CSG (flip inside vs. outside)
    pub fn inverse(&self) -> CSG {
        let mut csg = self.clone();
        for p in &mut csg.polygons {
            p.flip();
        }
        csg
    }

    /// Construct an axis-aligned cube, optional center and radius
    ///
    /// # Example
    ///
    /// ```
    /// let cube = CSG::cube(None);
    /// ```
    pub fn cube(options: Option<(&[f64; 3], &[f64; 3])>) -> CSG {
        let (center, radius) = match options {
            Some((c, r)) => (*c, *r),
            None => ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        };
        let c = Vector3::new(center[0], center[1], center[2]);
        let r = Vector3::new(radius[0], radius[1], radius[2]);

        let indices_and_normals = vec![
            (vec![0, 4, 6, 2], Vector3::new(-1.0, 0.0, 0.0)),
            (vec![1, 3, 7, 5], Vector3::new(1.0, 0.0, 0.0)),
            (vec![0, 1, 5, 4], Vector3::new(0.0, -1.0, 0.0)),
            (vec![2, 6, 7, 3], Vector3::new(0.0, 1.0, 0.0)),
            (vec![0, 2, 3, 1], Vector3::new(0.0, 0.0, -1.0)),
            (vec![4, 5, 7, 6], Vector3::new(0.0, 0.0, 1.0)),
        ];

        let mut polygons = Vec::new();
        for (idxs, n) in indices_and_normals {
            let mut verts = Vec::new();
            for i in idxs {
                // The bits of `i` pick +/- for x,y,z
                let vx = c.x + r.x * ((i & 1) as f64 * 2.0 - 1.0);
                let vy = c.y + r.y * (((i & 2) >> 1) as f64 * 2.0 - 1.0);
                let vz = c.z + r.z * (((i & 4) >> 2) as f64 * 2.0 - 1.0);
                verts.push(Vertex::new(Vector3::new(vx, vy, vz), n));
            }
            polygons.push(Polygon::new(verts, None));
        }

        CSG::from_polygons(polygons)
    }

    /// Construct a sphere with optional center, radius, slices, stacks
    ///
    /// # Example
    ///
    /// ```
    /// let sphere = CSG::sphere(None);
    /// ```
    pub fn sphere(options: Option<(&[f64; 3], f64, usize, usize)>) -> CSG {
        let (center, radius, slices, stacks) = match options {
            Some((c, r, sl, st)) => (*c, r, sl, st),
            None => ([0.0, 0.0, 0.0], 1.0, 16, 8),
        };

        let c = Vector3::new(center[0], center[1], center[2]);
        let mut polygons = Vec::new();

        // 2π is `std::f64::consts::TAU` in newer Rust,
        // but if that's not available, replace with `2.0 * std::f64::consts::PI`.
        for i in 0..slices {
            for j in 0..stacks {
                let mut vertices = Vec::new();

                let vertex = |theta: f64, phi: f64| {
                    let dir = Vector3::new(
                        theta.cos() * phi.sin(),
                        phi.cos(),
                        theta.sin() * phi.sin(),
                    );
                    Vertex::new(c + dir * radius, dir)
                };

                let t0 = i as f64 / slices as f64;
                let t1 = (i + 1) as f64 / slices as f64;
                let p0 = j as f64 / stacks as f64;
                let p1 = (j + 1) as f64 / stacks as f64;

                let theta0 = t0 * std::f64::consts::TAU; 
                let theta1 = t1 * std::f64::consts::TAU;
                let phi0 = p0 * std::f64::consts::PI;
                let phi1 = p1 * std::f64::consts::PI;

                vertices.push(vertex(theta0, phi0));
                if j > 0 {
                    vertices.push(vertex(theta1, phi0));
                }
                if j < stacks - 1 {
                    vertices.push(vertex(theta1, phi1));
                }
                vertices.push(vertex(theta0, phi1));

                polygons.push(Polygon::new(vertices, None));
            }
        }

        CSG::from_polygons(polygons)
    }

    /// Construct a cylinder with optional start, end, radius, slices
    ///
    /// # Example
    ///
    /// ```
    /// let cylinder = CSG::cylinder(None);
    /// ```
    pub fn cylinder(options: Option<(&[f64; 3], &[f64; 3], f64, usize)>) -> CSG {
        let (start, end, radius, slices) = match options {
            Some((s, e, r, sl)) => (*s, *e, r, sl),
            None => ([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], 1.0, 16),
        };

        let s = Vector3::new(start[0], start[1], start[2]);
        let e = Vector3::new(end[0], end[1], end[2]);
        let ray = e - s;
        let axis_z = ray.normalize();
        let is_y = axis_z.y.abs() > 0.5;

        // If axis_z is mostly aligned with Y, pick X; otherwise pick Y.
        let mut axis_x = if is_y {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        axis_x = axis_x.cross(&axis_z).normalize();
        let axis_y = axis_x.cross(&axis_z).normalize();

        let start_v = Vertex::new(s, -axis_z);
        let end_v = Vertex::new(e, axis_z);

        let mut polygons = Vec::new();

        let point = |stack: f64, slice: f64, normal_blend: f64| {
            let angle = slice * std::f64::consts::TAU;
            let out = axis_x * angle.cos() + axis_y * angle.sin();
            let pos = s + ray * stack + out * radius;
            // Blend outward normal with axis_z for the cap edges
            let normal = out * (1.0 - normal_blend.abs()) + axis_z * normal_blend;
            Vertex::new(pos, normal)
        };

        for i in 0..slices {
            let t0 = i as f64 / slices as f64;
            let t1 = (i + 1) as f64 / slices as f64;

            // bottom cap
            polygons.push(Polygon::new(
                vec![
                    start_v.clone(),
                    point(0.0, t0, -1.0),
                    point(0.0, t1, -1.0),
                ],
                None,
            ));

            // tube
            polygons.push(Polygon::new(
                vec![
                    point(0.0, t1, 0.0),
                    point(0.0, t0, 0.0),
                    point(1.0, t0, 0.0),
                    point(1.0, t1, 0.0),
                ],
                None,
            ));

            // top cap
            polygons.push(Polygon::new(
                vec![
                    end_v.clone(),
                    point(1.0, t1, 1.0),
                    point(1.0, t0, 1.0),
                ],
                None,
            ));
        }

        CSG::from_polygons(polygons)
    }

    // ----------------------------------------------------------
    //   Export to ASCII STL
    // ----------------------------------------------------------

    /// Convert this CSG to an **ASCII STL** string with the given `name`.
    ///
    /// ```
    /// let csg = CSG::cube(None);
    /// let stl_text = csg.to_stl("my_solid");
    /// println!("{}", stl_text);
    /// ```
    pub fn to_stl(&self, name: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("solid {}\n", name));

        for poly in &self.polygons {
            // Use the polygon plane's normal for the facet normal (normalized).
            let normal = poly.plane.normal.normalize();
            let triangles = poly.triangulate();

            for tri in triangles {
                out.push_str(&format!(
                    "  facet normal {:.6} {:.6} {:.6}\n",
                    normal.x, normal.y, normal.z
                ));
                out.push_str("    outer loop\n");
                for vertex in &tri {
                    out.push_str(&format!(
                        "      vertex {:.6} {:.6} {:.6}\n",
                        vertex.pos.x, vertex.pos.y, vertex.pos.z
                    ));
                }
                out.push_str("    endloop\n");
                out.push_str("  endfacet\n");
            }
        }

        out.push_str(&format!("endsolid {}\n", name));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    // --------------------------------------------------------
    //   Helpers
    // --------------------------------------------------------

    /// Returns the approximate bounding box `[min_x, min_y, min_z, max_x, max_y, max_z]`
    /// for a set of polygons.
    fn bounding_box(polygons: &[Polygon]) -> [f64; 6] {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut min_z = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut max_z = f64::MIN;

        for poly in polygons {
            for v in &poly.vertices {
                let p = v.pos;
                if p.x < min_x {
                    min_x = p.x;
                }
                if p.y < min_y {
                    min_y = p.y;
                }
                if p.z < min_z {
                    min_z = p.z;
                }
                if p.x > max_x {
                    max_x = p.x;
                }
                if p.y > max_y {
                    max_y = p.y;
                }
                if p.z > max_z {
                    max_z = p.z;
                }
            }
        }

        [min_x, min_y, min_z, max_x, max_y, max_z]
    }

    /// Quick helper to compare floating-point results with an acceptable tolerance.
    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // --------------------------------------------------------
    //   Vertex Tests
    // --------------------------------------------------------

    #[test]
    fn test_vertex_interpolate() {
        let v1 = Vertex::new(Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));
        let v2 = Vertex::new(Vector3::new(10.0, 10.0, 10.0), Vector3::new(0.0, 1.0, 0.0));

        let v_mid = v1.interpolate(&v2, 0.5);
        assert!(approx_eq(v_mid.pos.x, 5.0, 1e-8));
        assert!(approx_eq(v_mid.pos.y, 5.0, 1e-8));
        assert!(approx_eq(v_mid.pos.z, 5.0, 1e-8));

        // Normal should also interpolate
        assert!(approx_eq(v_mid.normal.x, 0.5, 1e-8));
        assert!(approx_eq(v_mid.normal.y, 0.5, 1e-8));
        assert!(approx_eq(v_mid.normal.z, 0.0, 1e-8));
    }

    #[test]
    fn test_vertex_flip() {
        let mut v = Vertex::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 0.0, 0.0));
        v.flip();
        assert_eq!(v.pos, Vector3::new(1.0, 2.0, 3.0));
        // Normal should be negated
        assert_eq!(v.normal, Vector3::new(-1.0, 0.0, 0.0));
    }

    // --------------------------------------------------------
    //   Polygon Tests
    // --------------------------------------------------------

    #[test]
    fn test_polygon_construction() {
        let v1 = Vertex::new(Vector3::new(0.0, 0.0, 0.0), Vector3::y());
        let v2 = Vertex::new(Vector3::new(1.0, 0.0, 0.0), Vector3::y());
        let v3 = Vertex::new(Vector3::new(0.0, 1.0, 0.0), Vector3::y());

        let poly = Polygon::new(vec![v1.clone(), v2.clone(), v3.clone()], None);
        assert_eq!(poly.vertices.len(), 3);
        // Plane should be defined by these three points
        assert!(approx_eq(poly.plane.normal.dot(&Vector3::y()), 1.0, 1e-8));
    }

    #[test]
    fn test_polygon_flip() {
        let v1 = Vertex::new(Vector3::new(0.0, 0.0, 0.0), Vector3::y());
        let v2 = Vertex::new(Vector3::new(1.0, 0.0, 0.0), Vector3::y());
        let v3 = Vertex::new(Vector3::new(0.0, 1.0, 0.0), Vector3::y());
        let mut poly = Polygon::new(vec![v1, v2, v3], None);

        let original_normal = poly.plane.normal;
        poly.flip();
        // The vertex order should be reversed
        assert_eq!(poly.vertices.len(), 3);
        // The plane’s normal should be reversed
        let flipped_normal = poly.plane.normal;
        assert_eq!(flipped_normal, -original_normal);
    }

    #[test]
    fn test_polygon_triangulate() {
        let v1 = Vertex::new(Vector3::new(0.0, 0.0, 0.0), Vector3::z());
        let v2 = Vertex::new(Vector3::new(1.0, 0.0, 0.0), Vector3::z());
        let v3 = Vertex::new(Vector3::new(1.0, 1.0, 0.0), Vector3::z());
        let v4 = Vertex::new(Vector3::new(0.0, 1.0, 0.0), Vector3::z());
        let poly = Polygon::new(vec![v1, v2, v3, v4], None);

        let triangles = poly.triangulate();
        assert_eq!(triangles.len(), 2, "A quad should triangulate into 2 triangles");
    }

    // --------------------------------------------------------
    //   CSG: Basic Shape Generation
    // --------------------------------------------------------

    #[test]
    fn test_csg_cube() {
        // Default cube is centered at (0,0,0) with radius (1,1,1)
        let cube = CSG::cube(None);
        let polys = cube.to_polygons();
        assert_eq!(polys.len(), 6, "Cube should have 6 faces (polygons)");

        // Check bounding box => from (-1,-1,-1) to (1,1,1)
        let bb = bounding_box(polys);
        for &val in &bb[..3] {
            assert!(approx_eq(val, -1.0, 1e-8));
        }
        for &val in &bb[3..] {
            assert!(approx_eq(val, 1.0, 1e-8));
        }
    }

    #[test]
    fn test_csg_sphere() {
        // Default sphere => radius=1, slices=16, stacks=8
        let sphere = CSG::sphere(None);
        let polys = sphere.to_polygons();
        assert!(!polys.is_empty(), "Sphere should generate polygons");

        let bb = bounding_box(polys);
        // Should roughly be [-1, -1, -1, 1, 1, 1]
        assert!(approx_eq(bb[0], -1.0, 1e-1));
        assert!(approx_eq(bb[1], -1.0, 1e-1));
        assert!(approx_eq(bb[2], -1.0, 1e-1));
        assert!(approx_eq(bb[3],  1.0, 1e-1));
        assert!(approx_eq(bb[4],  1.0, 1e-1));
        assert!(approx_eq(bb[5],  1.0, 1e-1));
    }

    #[test]
    fn test_csg_cylinder() {
        // Default cylinder => from (0,-1,0) to (0,1,0) with radius=1
        let cylinder = CSG::cylinder(None);
        let polys = cylinder.to_polygons();
        assert!(!polys.is_empty(), "Cylinder should generate polygons");

        let bb = bounding_box(polys);
        // Expect x in [-1,1], y in [-1,1], z near 0 for the side, but also check caps
        assert!(approx_eq(bb[0], -1.0, 1e-8));
        assert!(approx_eq(bb[3],  1.0, 1e-8));
        assert!(approx_eq(bb[1], -1.0, 1e-8));
        assert!(approx_eq(bb[4],  1.0, 1e-8));
        // z should be roughly within [-1e-8, +1e-8] since center is z=0
        assert!(bb[2] >= -1e-7 && bb[5] <= 1e-7);
    }

    // --------------------------------------------------------
    //   CSG: Operations (union, subtract, intersect)
    // --------------------------------------------------------

    #[test]
    fn test_csg_union() {
        let cube1 = CSG::cube(None); // from -1 to +1 in all coords
        let cube2 = CSG::cube(Some((&[0.5, 0.5, 0.5], &[1.0, 1.0, 1.0])));
        // The second cube is centered at (0.5,0.5,0.5)

        let union_csg = cube1.union(&cube2);
        let polys = union_csg.to_polygons();
        assert!(!polys.is_empty(), "Union of two cubes should produce polygons");

        // Check bounding box => it should now at least range from -1 to (0.5+1) = 1.5
        let bb = bounding_box(polys);
        assert!(approx_eq(bb[0], -1.0, 1e-8));
        assert!(approx_eq(bb[1], -1.0, 1e-8));
        assert!(approx_eq(bb[2], -1.0, 1e-8));
        assert!(approx_eq(bb[3], 1.5, 1e-8));
        assert!(approx_eq(bb[4], 1.5, 1e-8));
        assert!(approx_eq(bb[5], 1.5, 1e-8));
    }

    #[test]
    fn test_csg_subtract() {
        // Subtract a smaller cube from a bigger one
        let big_cube = CSG::cube(Some((&[0.0, 0.0, 0.0], &[2.0, 2.0, 2.0]))); // radius=2 => spans [-2,2]
        let small_cube = CSG::cube(None); // radius=1 => spans [-1,1]

        let result = big_cube.subtract(&small_cube);
        let polys = result.to_polygons();
        assert!(!polys.is_empty(), "Subtracting a smaller cube should leave polygons");

        // Check bounding box => should still be [-2,-2,-2, 2,2,2], but with a chunk removed
        let bb = bounding_box(polys);
        // At least the bounding box remains the same
        assert!(approx_eq(bb[0], -2.0, 1e-8));
        assert!(approx_eq(bb[3],  2.0, 1e-8));
    }

    #[test]
    fn test_csg_intersect() {
        let sphere = CSG::sphere(None);
        let cube = CSG::cube(None);

        let intersection = sphere.intersect(&cube);
        let polys = intersection.to_polygons();
        assert!(!polys.is_empty(), "Sphere ∩ Cube should produce a shape (the portion of the sphere inside the cube)");

        // Check bounding box => intersection is roughly a sphere clipped to [-1,1]^3
        let bb = bounding_box(polys);
        // Should be a region inside the [-1,1] box
        for &val in &bb[..3] {
            assert!(val >= -1.0 - 1e-1);
        }
        for &val in &bb[3..] {
            assert!(val <= 1.0 + 1e-1);
        }
    }

    #[test]
    fn test_csg_inverse() {
        let cube = CSG::cube(None);
        let inv_cube = cube.inverse();
        assert_eq!(inv_cube.to_polygons().len(), cube.to_polygons().len(), 
            "Inverse should keep the same polygon count, but flip them");
    }

    // --------------------------------------------------------
    //   CSG: STL Export
    // --------------------------------------------------------

    #[test]
    fn test_to_stl() {
        let cube = CSG::cube(None);
        let stl_str = cube.to_stl("test_cube");
        // Basic checks
        assert!(stl_str.contains("solid test_cube"));
        assert!(stl_str.contains("endsolid test_cube"));

        // Should contain some facet normals
        assert!(stl_str.contains("facet normal"));
        // Should contain some vertex lines
        assert!(stl_str.contains("vertex"));
    }

    // --------------------------------------------------------
    //   Node & Clipping Tests
    //   (Optional: these get more into internal details)
    // --------------------------------------------------------

    #[test]
    fn test_node_clip_polygons() {
        // Build a simple BSP from a cube
        let cube = CSG::cube(None);
        let mut node = Node::new(cube.polygons.clone());
        // Now clip the same polygons => we should get them back (none are inside)
        let clipped = node.clip_polygons(&cube.polygons);
        assert_eq!(clipped.len(), cube.polygons.len());
    }

    #[test]
    fn test_node_invert() {
        let cube = CSG::cube(None);
        let mut node = Node::new(cube.polygons.clone());
        let original_count = node.polygons.len();
        // Invert them
        node.invert();
        // We shouldn’t lose polygons by inverting
        assert_eq!(node.polygons.len(), original_count);
        // If we invert back, we should get the same geometry
        node.invert();
        assert_eq!(node.polygons.len(), original_count);
    }
}


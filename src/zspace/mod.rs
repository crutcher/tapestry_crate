//! Z-space is a vector space with a basis of vectors with integer coordinates.

use ops::{Add, Div, Mul, Rem, Sub};
use std::fmt;
use std::ops;

use ndarray;

/// A trait for types that have a dimensionality.
pub trait ZDim {
    /// Return the dimensionality of the object.
    fn ndim(&self) -> usize;
}

impl ZDim for ndarray::Array<i32, ndarray::Ix1> {
    /// Get the number of dimensions of the array.
    fn ndim(&self) -> usize {
        self.shape()[0]
    }
}

impl ZDim for Vec<i32> {
    /// Get the number of dimensions of the vector.
    fn ndim(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod zdim_tests {
    use super::*;

    #[test]
    fn test_ndim() {
        let a: ndarray::Array1<i32> = ndarray::Array1::from_vec(vec![1, 2, 3]);
        assert_eq!(ZDim::ndim(&a), 3);
        assert_eq!(a[0], 1);

        let b = vec![1, 2, 3];
        assert_eq!(ZDim::ndim(&b), 3);
        assert_eq!(b[0], 1);
    }
}

macro_rules! assert_same_zdim {
    // The `tt` (token tree) designator is used for
    // operators and tokens.
    ($a:expr, $b:expr, $func:ident, $op:tt) => {
        assert!(
            ZDim::ndim(&$a) == ZDim::ndim(&$b),
            "{:?}: dimension mismatch: {:?} {:?} {:?}",
            stringify!($func),
            (ZDim::ndim(&$a),),
            stringify!($op),
            (ZDim::ndim(&$b),)
        );
    };
}

/// A ZPoint is an immutable point in Z-space.
#[derive(Debug, Clone)]
pub struct ZPoint {
    pub coords: ndarray::Array1<i32>,
}

impl fmt::Display for ZPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements: Vec<String> = self.coords.iter().map(|x| x.to_string()).collect();
        write!(f, "Z[{}]", elements.join(", "))
    }
}

#[cfg(test)]
mod test_zpoint_display {
    use super::*;

    #[test]
    fn test_display() {
        let a = ZPoint::from(vec![1, 2, 3]);
        assert_eq!(format!("{}", a), "Z[1, 2, 3]");
    }
}

use std::cmp::Ordering;

impl PartialOrd for ZPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        assert_same_zdim!(self, other, partial_cmp, <);
        if self.ndim() == 0 {
            return Some(Ordering::Equal);
        }

        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a.partial_cmp(b))
            .reduce(|a, b| match (a, b) {
                (None, _) => None,
                (_, None) => None,
                (Some(Ordering::Equal), _) => b,
                (Some(Ordering::Less), Some(Ordering::Greater)) => None,
                (Some(Ordering::Less), _) => Some(Ordering::Less),
                (Some(Ordering::Greater), Some(Ordering::Less)) => None,
                (Some(Ordering::Greater), _) => Some(Ordering::Greater),
            })?
    }
}

impl PartialEq for ZPoint {
    fn eq(&self, other: &Self) -> bool {
        assert_same_zdim!(self, other, eq, ==);
        self.coords.iter().eq(other.coords.iter())
    }
}
impl Eq for ZPoint {}

#[cfg(test)]
mod test_zpoint_ordering {
    use super::*;

    #[test]
    fn test_partial() {
        let z = ZPoint::zeros(3);
        assert_eq!(z < z, false);
        assert_eq!(z <= z, true);
        assert_eq!(z > z, false);
        assert_eq!(z >= z, true);
        assert_eq!(z == z, true);
        assert_eq!(z != z, false);

        let a = ZPoint::from(vec![1, 2, 3]);
        assert_eq!(z < a, true);
        assert_eq!(z <= a, true);
        assert_eq!(z > a, false);
        assert_eq!(z >= a, false);
        assert_eq!(z != a, true);
        assert_eq!(z == a, false);

        let b = ZPoint::from(vec![1, 1, 4]);
        assert_eq!(a < b, false);
        assert_eq!(a <= b, false);
        assert_eq!(a > b, false);
        assert_eq!(a >= b, false);
        assert_eq!(a != b, true);
        assert_eq!(a == b, false);
    }
}

impl ZDim for ZPoint {
    /// Get the number of dimensions of the ZPoint.
    fn ndim(&self) -> usize {
        self.coords.len()
    }
}

/// A trait for types that can be converted to a ZPoint.
pub trait ZPointSource {
    /// Convert the object to a ZPoint.
    fn new_zpoint(self) -> ZPoint;
}

impl ZPointSource for ZPoint {
    fn new_zpoint(self) -> ZPoint {
        self
    }
}

impl ZPointSource for &ZPoint {
    fn new_zpoint(self) -> ZPoint {
        self.clone()
    }
}

impl ZPointSource for ndarray::Array1<i32> {
    fn new_zpoint(self) -> ZPoint {
        ZPoint { coords: self }
    }
}

impl ZPointSource for &ndarray::Array1<i32> {
    fn new_zpoint(self) -> ZPoint {
        ZPoint {
            coords: self.clone(),
        }
    }
}

impl ZPointSource for Vec<i32> {
    fn new_zpoint(self) -> ZPoint {
        ZPoint {
            coords: ndarray::Array1::from_vec(self),
        }
    }
}

impl ZPointSource for &Vec<i32> {
    fn new_zpoint(self) -> ZPoint {
        ZPoint {
            coords: ndarray::Array1::from_vec(self.clone()),
        }
    }
}

impl ZDim for &ZPoint {
    /// Get the number of dimensions of the ZPoint.
    fn ndim(&self) -> usize {
        self.coords.len()
    }
}

impl ops::Index<usize> for ZPoint {
    type Output = i32;

    /// Index operator for ZPoint.
    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

/// A point in Z-space.
/// Z-space is a vector space with a basis of vectors with integer coordinates.
impl ZPoint {
    /// Create a new ZPoint from a ZPointSource.
    pub fn from(source: impl ZPointSource) -> Self {
        source.new_zpoint()
    }
}

#[cfg(test)]
mod test_from {
    use super::*;

    #[test]
    fn test_from_vec() {
        let expected = ndarray::Array1::from_vec(vec![1, 2, 3]);

        // move Vec<i32>
        assert_eq!(ZPoint::from(vec![1, 2, 3]).coords, &expected);

        // borrow Vec<i32>
        assert_eq!(ZPoint::from(&vec![1, 2, 3]).coords, &expected);
    }

    #[test]
    fn test_from_ndarray() {
        let expected = ndarray::Array1::from_vec(vec![1, 2, 3]);

        // move Vec<i32>
        assert_eq!(
            ZPoint::from(ndarray::Array1::from_vec(vec![1, 2, 3])).coords,
            &expected
        );

        // borrow Vec<i32>
        assert_eq!(
            ZPoint::from(&ndarray::Array1::from_vec(vec![1, 2, 3])).coords,
            &expected
        );
    }
}

impl ZPoint {
    /// Create a new ZPoint of zeros with the given dimensionality.
    pub fn zeros(ndim: usize) -> Self {
        ZPoint {
            coords: ndarray::Array1::zeros(ndim),
        }
    }

    /// Create a new ZPoint of zeros with the same dimensionality as the given ZPoint.
    pub fn zeros_like(other: &ZPoint) -> Self {
        ZPoint::zeros(other.ndim())
    }

    /// Create a new ZPoint of ones with the given dimensionality.
    pub fn ones(ndim: usize) -> Self {
        ZPoint {
            coords: ndarray::Array1::ones(ndim),
        }
    }

    /// Create a new ZPoint of ones with the same dimensionality as the given ZPoint.
    pub fn ones_like(other: &ZPoint) -> Self {
        ZPoint::ones(other.ndim())
    }

    /// Create a new ZPoint with the given dimensionality and value.
    pub fn full(ndim: usize, value: i32) -> Self {
        ZPoint {
            coords: ndarray::Array1::from_elem(ndim, value),
        }
    }

    /// Create a new ZPoint with the same dimensionality as the given ZPoint and value.
    pub fn full_like(other: &ZPoint, value: i32) -> Self {
        ZPoint::full(other.ndim(), value)
    }
}

macro_rules! uniop {
    ($bound:ident, $op:tt, $method:ident) => {
        impl ops::$bound for &ZPoint {
            type Output = ZPoint;

            fn $method(self) -> ZPoint {
                ZPoint {
                    coords: $op & self.coords,
                }
            }
        }
    };
}

uniop!(Neg, -, neg);

macro_rules! binop (
    ($trt:ident, $op:tt, $mth:ident) => (
        // Implementation of binary operator `op` for `ZPoint`;
        // covering the cases `ZPoint op ZPoint`, `&ZPoint op ZPoint`, `ZPoint op &ZPoint`:
        impl $trt<&ZPoint> for &ZPoint {
            type Output = ZPoint;
            fn $mth(self, other: &ZPoint) -> Self::Output {
                assert_same_zdim!(self, other, $mth, $op);
                ZPoint {
                    coords: &self.coords $op &other.coords,
                }
            }
        }

        impl $trt<&Vec<i32>> for &ZPoint {
            type Output = ZPoint;
            fn $mth(self, other: &Vec<i32>) -> Self::Output {
                self $op &ZPoint::from(other.clone())
            }
        }

        impl $trt<&ZPoint> for &Vec<i32> {
            type Output = ZPoint;
            fn $mth(self, other: &ZPoint) -> Self::Output {
                &ZPoint::from(self.clone()) $op other
            }
        }

        impl $trt<&ndarray::Array1<i32>> for &ZPoint {
            type Output = ZPoint;
            fn $mth(self, other: &ndarray::Array1<i32>) -> Self::Output {
                self $op &ZPoint::from(other.clone())
            }
        }
        impl $trt<&ZPoint> for &ndarray::Array1<i32> {
            type Output = ZPoint;
            fn $mth(self, other: &ZPoint) -> Self::Output {
                &ZPoint::from(self.clone()) $op other
            }
        }

        impl<'a> $trt<i32> for &'a ZPoint
        {
            type Output = ZPoint;

            fn $mth(self, other: i32) -> Self::Output {
                ZPoint {
                    coords: &self.coords $op other,
                }
            }
        }

        impl<'a> $trt<&'a ZPoint> for i32
        {
            type Output = ZPoint;

            fn $mth(self, other: &'a ZPoint) -> Self::Output {
                ZPoint {
                    coords: self $op &other.coords,
                }
            }
        }

    );
);

binop!(Add, +, add);
binop!(Sub, -, sub);
binop!(Mul, *, mul);
binop!(Rem, %, rem);
binop!(Div, /, div);

#[cfg(test)]
mod zpoint_tests {
    use super::*;

    #[test]
    fn test_clone() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = point1.clone();
        assert_eq!(point1, point2);
    }

    #[test]
    fn test_zeros() {
        let point = ZPoint::zeros(3);
        assert_eq!(point.ndim(), 3);
        assert_eq!(point, ZPoint::from(vec![0, 0, 0]));
    }

    #[test]
    fn test_zeros_like() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::zeros_like(&point1);
        assert_eq!(point2.ndim(), 3);
        assert_eq!(point2, ZPoint::from(vec![0, 0, 0]));
    }

    #[test]
    fn test_ones() {
        let point = ZPoint::ones(3);
        assert_eq!(point.ndim(), 3);
        assert_eq!(point, ZPoint::from(vec![1, 1, 1]));
    }

    #[test]
    fn test_ones_like() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::ones_like(&point1);
        assert_eq!(point2.ndim(), 3);
        assert_eq!(point2, ZPoint::from(vec![1, 1, 1]));
    }

    #[test]
    fn test_full() {
        let point = ZPoint::full(3, 5);
        assert_eq!(point.ndim(), 3);
        assert_eq!(point, ZPoint::from(vec![5, 5, 5]));
    }

    #[test]
    fn test_full_like() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::full_like(&point1, 5);
        assert_eq!(point2.ndim(), 3);
        assert_eq!(point2, ZPoint::from(vec![5, 5, 5]));
    }

    #[test]
    fn test_from_vec() {
        let point = ZPoint::from(vec![1, 2, 3]);
        assert_eq!(point.ndim(), 3);
        assert_eq!(point[0], 1);
        assert_eq!(point[1], 2);
        assert_eq!(point[2], 3);
    }

    #[test]
    fn test_from_ndarray() {
        let point = ZPoint::from(ndarray::array![1, 2, 3]);
        assert_eq!(point.ndim(), 3);
        assert_eq!(point[0], 1);
        assert_eq!(point[1], 2);
        assert_eq!(point[2], 3);
    }

    #[test]
    fn test_add() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::from(vec![4, 5, 6]);
        assert_eq!((&point1 + &point2), ZPoint::from(vec![5, 7, 9]));

        // scalars
        assert_eq!(&point1 + 1, ZPoint::from(vec![2, 3, 4]));
        assert_eq!(1 + &point1, ZPoint::from(vec![2, 3, 4]));

        // ndarrays
        assert_eq!(
            &point1 + &ndarray::array![1, 2, 3],
            ZPoint::from(vec![2, 4, 6])
        );
        assert_eq!(
            &ndarray::array![1, 2, 3] + &point1,
            ZPoint::from(vec![2, 4, 6])
        );

        // vec
        assert_eq!(&point1 + &vec![1, 2, 3], ZPoint::from(vec![2, 4, 6]));
        assert_eq!(&vec![1, 2, 3] + &point1, ZPoint::from(vec![2, 4, 6]));
    }

    #[test]
    fn test_sub() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::from(vec![4, 5, 6]);
        assert_eq!((&point1 - &point2), ZPoint::from(vec![-3, -3, -3]));

        // scalars
        assert_eq!(&point1 - 1, ZPoint::from(vec![0, 1, 2]));
        assert_eq!(1 - &point1, ZPoint::from(vec![0, -1, -2]));

        // ndarrays
        assert_eq!(
            &point1 - &ndarray::array![1, 2, 3],
            ZPoint::from(vec![0, 0, 0])
        );
        assert_eq!(
            &ndarray::array![1, 2, 3] - &point1,
            ZPoint::from(vec![0, 0, 0])
        );

        // vec
        assert_eq!(&point1 - &vec![1, 2, 3], ZPoint::from(vec![0, 0, 0]));
        assert_eq!(&vec![1, 2, 3] - &point1, ZPoint::from(vec![0, 0, 0]));
    }

    #[test]
    fn test_mul() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::from(vec![4, 5, 6]);
        assert_eq!((&point1 * &point2), ZPoint::from(vec![4, 10, 18]));

        // scalars
        assert_eq!(&point1 * 2, ZPoint::from(vec![2, 4, 6]));
        assert_eq!(2 * &point1, ZPoint::from(vec![2, 4, 6]));

        // ndarrays
        assert_eq!(
            &point1 * &ndarray::array![1, 2, 3],
            ZPoint::from(vec![1, 4, 9])
        );
        assert_eq!(
            &ndarray::array![1, 2, 3] * &point1,
            ZPoint::from(vec![1, 4, 9])
        );

        // vec
        assert_eq!(&point1 * &vec![1, 2, 3], ZPoint::from(vec![1, 4, 9]));
        assert_eq!(&vec![1, 2, 3] * &point1, ZPoint::from(vec![1, 4, 9]));
    }

    #[test]
    fn test_div() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::from(vec![4, 5, 6]);
        assert_eq!((&point1 / &point2), ZPoint::from(vec![0, 0, 0]));

        // scalars
        assert_eq!(&point1 / 2, ZPoint::from(vec![0, 1, 1]));
        assert_eq!(2 / &point1, ZPoint::from(vec![2, 1, 0]));

        // ndarrays
        assert_eq!(
            &point1 / &ndarray::array![1, 2, 3],
            ZPoint::from(vec![1, 1, 1])
        );
        assert_eq!(
            &ndarray::array![1, 2, 3] / &point1,
            ZPoint::from(vec![1, 1, 1])
        );

        // vec
        assert_eq!(&point1 / &vec![1, 2, 3], ZPoint::from(vec![1, 1, 1]));
        assert_eq!(&vec![1, 2, 3] / &point1, ZPoint::from(vec![1, 1, 1]));
    }

    #[test]
    fn test_rem() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = ZPoint::from(vec![4, 5, 6]);
        assert_eq!((&point1 % &point2), ZPoint::from(vec![1, 2, 3]));

        // scalars
        assert_eq!(&point1 % 2, ZPoint::from(vec![1, 0, 1]));
        assert_eq!(2 % &point1, ZPoint::from(vec![0, 0, 2]));

        // ndarrays
        assert_eq!(
            &point1 % &ndarray::array![1, 2, 3],
            ZPoint::from(vec![0, 0, 0])
        );
        assert_eq!(
            &ndarray::array![1, 2, 3] % &point1,
            ZPoint::from(vec![0, 0, 0])
        );

        // vec
        assert_eq!(&point1 % &vec![1, 2, 3], ZPoint::from(vec![0, 0, 0]));
        assert_eq!(&vec![1, 2, 3] % &point1, ZPoint::from(vec![0, 0, 0]));
    }

    #[test]
    fn test_neg() {
        let point1 = ZPoint::from(vec![1, 2, 3]);
        let point2 = -&point1;
        assert_eq!(point2, ZPoint::from(vec![-1, -2, -3]));
    }
}

/// A ZRange is a rectangular prism in ZSpace, defined over `[start, end)`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZRange {
    start: ZPoint,
    end: ZPoint,
}

impl fmt::Display for ZRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pairs: Vec<String> = self
            .start
            .coords
            .iter()
            .zip(self.end.coords.iter())
            .map(|(a, b)| format!("{}:{}", a, b))
            .collect();

        write!(f, "ZR[{}]", pairs.join(", "))
    }
}

#[cfg(test)]
mod test_zrange_display {
    use super::*;

    #[test]
    fn test_display() {
        let range = ZRange::between(&ZPoint::from(vec![1, 2, 3]), &ZPoint::from(vec![4, 5, 6]));
        assert_eq!(format!("{}", range), "ZR[1:4, 2:5, 3:6]");
    }
}

impl ZDim for ZRange {
    /// Get the number of dimensions of the ZRange
    fn ndim(&self) -> usize {
        return self.start.ndim();
    }
}

impl ZRange {
    pub fn from_shape(shape: impl ZPointSource) -> Self {
        let source = shape.new_zpoint();
        ZRange::between(&ZPoint::zeros_like(&source), &source)
    }

    pub fn between(start: impl ZPointSource, end: impl ZPointSource) -> Self {
        let start = start.new_zpoint();
        let end = end.new_zpoint();

        assert_same_zdim!(start, end, between, ..);

        assert!(
            start
                .coords
                .iter()
                .zip(end.coords.iter())
                .all(|(s, e)| { s <= e }),
            "start must be less than or equal to end: {:?} <= {:?}",
            start.coords,
            end.coords
        );

        ZRange {
            start: start.clone(),
            end: end.clone(),
        }
    }

    pub fn translate(&self, offset: impl ZPointSource) -> Self {
        let offset = offset.new_zpoint();
        assert_same_zdim!(self.start, offset, translate, ..);
        ZRange::between(&self.start + &offset, &self.end + &offset)
    }

    pub fn shape(&self) -> ZPoint {
        return &self.end - &self.start;
    }

    pub fn size(&self) -> usize {
        self.shape().coords.product() as usize
    }
}

#[cfg(test)]
mod zrange_tests {
    use super::*;

    #[test]
    fn test_from_zpoint_shape() {
        let shape = ZPoint::from(vec![1, 2, 3]);
        let range = ZRange::from_shape(&shape);
        assert_eq!(range.start, ZPoint::from(vec![0, 0, 0]));
        assert_eq!(range.end, shape);
    }

    #[test]
    fn test_from_vec_shape() {
        let shape = ZPoint::from(vec![1, 2, 3]);
        let range = ZRange::from_shape(vec![1, 2, 3]);
        assert_eq!(range.start, ZPoint::from(vec![0, 0, 0]));
        assert_eq!(range.end, shape);
    }

    #[test]
    fn test_from_ndarray_shape() {
        let shape = ZPoint::from(vec![1, 2, 3]);
        let range = ZRange::from_shape(ndarray::array![1, 2, 3]);
        assert_eq!(range.start, ZPoint::from(vec![0, 0, 0]));
        assert_eq!(range.end, shape);
    }

    #[test]
    fn test_between() {
        let start = ZPoint::from(vec![1, 2, 3]);
        let end = ZPoint::from(vec![4, 5, 6]);
        let range = ZRange::between(&start, &end);
        assert_eq!(range.start, start);
        assert_eq!(range.end, end);
    }

    #[test]
    fn test_translate() {
        let start = ZPoint::from(vec![1, 2, 3]);
        let end = ZPoint::from(vec![4, 5, 6]);
        let range = ZRange::between(&start, &end);
        let offset = ZPoint::from(vec![1, 2, 3]);
        let translated = range.translate(&offset);
        assert_eq!(translated.start, &start + &offset);
        assert_eq!(translated.end, &end + &offset);
    }

    #[test]
    #[should_panic]
    fn test_between_panic() {
        ZRange::between(&ZPoint::zeros(2), &ZPoint::ones(3));
    }

    #[test]
    fn test_ndim() {
        let start = ZPoint::from(vec![1, 2, 3]);
        let end = ZPoint::from(vec![4, 5, 6]);
        let range = ZRange::between(&start, &end);
        assert_eq!(range.ndim(), 3);
    }

    #[test]
    fn test_shape_size() {
        let start = ZPoint::from(vec![1, 2, 3]);
        let end = ZPoint::from(vec![4, 5, 6]);
        let range = ZRange::between(&start, &end);
        assert_eq!(range.shape(), ZPoint::from(vec![3, 3, 3]));
        assert_eq!(range.size(), 27);
    }
}

impl ZRange {
    pub fn contains(&self, point: impl ZPointSource) -> bool {
        let point = point.new_zpoint();
        assert_same_zdim!(self.start, point, contains, ..);
        self.start
            .coords
            .iter()
            .zip(point.coords.iter())
            .all(|(s, p)| s <= p)
            && self
                .end
                .coords
                .iter()
                .zip(point.coords.iter())
                .all(|(e, p)| e > p)
    }
}

#[cfg(test)]
mod test_contains {
    use super::*;

    #[test]
    fn test_contains() {
        let range = ZRange::between(&ZPoint::from(vec![1, 2, 3]), &ZPoint::from(vec![4, 5, 6]));
        assert!(range.contains(&ZPoint::from(vec![2, 3, 4])));
        assert!(!range.contains(&ZPoint::from(vec![0, 3, 4])));
        assert!(!range.contains(&ZPoint::from(vec![2, 6, 4])));
        assert!(!range.contains(&ZPoint::from(vec![2, 3, 7])));
    }
}

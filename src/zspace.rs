use ndarray;
use std::ops;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZPoint {
    coords: ndarray::Array1<i32>,
}

impl ZPoint {
    pub fn zeros(ndim: usize) -> Self {
        ZPoint { coords: ndarray::Array1::zeros(ndim) }
    }

    pub fn zeros_like(other: &ZPoint) -> Self {
        ZPoint { coords: ndarray::Array1::zeros(other.ndim()) }
    }

    pub fn ones(ndim: usize) -> Self {
        ZPoint { coords: ndarray::Array1::ones(ndim) }
    }

    pub fn ones_like(other: &ZPoint) -> Self {
        ZPoint { coords: ndarray::Array1::ones(other.ndim()) }
    }

    pub fn from_ndarray(coords: ndarray::Array1<i32>) -> Self {
        ZPoint { coords }
    }

    pub fn from_vec(coords: Vec<i32>) -> Self {
        ZPoint { coords: ndarray::Array1::from_vec(coords) }
    }

    pub fn ndim(&self) -> usize {
        return self.coords.len();
    }
}

impl ops::Index<usize> for ZPoint {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

macro_rules! assert_equal_ndim {
    // The `tt` (token tree) designator is used for
    // operators and tokens.
    ($a:expr, $b:expr, $func:ident, $op:tt) => {
        assert!($a.len() == $b.len(),
                "{:?}: dimension mismatch: {:?} {:?} {:?}",
                stringify!($func),
                ($a.len(),),
                stringify!($op),
                ($b.len(),));
    };
}

macro_rules! uniop {
    ($bound:ident, $op:tt, $method:ident) => {
        impl ops::$bound for &ZPoint {
            type Output = ZPoint;

            fn $method(self) -> ZPoint {
                ZPoint { coords: $op &self.coords }
            }
        }

    };
}

uniop!(Neg, -, neg);

macro_rules! binop {
    ($bound:ident, $op:tt, $method:ident) => {
        impl ops::$bound for &ZPoint {
            type Output = ZPoint;

            fn $method(self, other: Self) -> ZPoint {
                assert_equal_ndim!(self.coords, self.coords, $method, $op);
                ZPoint { coords: &self.coords $op &other.coords }
            }
        }

    };
}

binop!(Add, +, add);
binop!(Sub, -, sub);
binop!(Mul, *, mul);
binop!(Rem, %, rem);


#[cfg(test)]
mod zpoint_tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let point = ZPoint::zeros(3);
        assert_eq!(point.ndim(), 3);
    }

    #[test]
    fn test_ones() {
        let point = ZPoint::ones(3);
        assert_eq!(point.ndim(), 3);
    }

    #[test]
    fn test_from_vec() {
        let point = ZPoint::from_vec(vec![1, 2, 3]);
        assert_eq!(point.ndim(), 3);
    }

    #[test]
    fn test_from_ndarray() {
        let point = ZPoint::from_ndarray(ndarray::array![1, 2, 3]);
        assert_eq!(point.ndim(), 3);
    }

    #[test]
    fn test_index() {
        let point = ZPoint::from_vec(vec![1, 2, 3]);
        assert_eq!(point[0], 1);
        assert_eq!(point[1], 2);
        assert_eq!(point[2], 3);
    }

    #[test]
    fn test_ndim() {
        let point = ZPoint::from_vec(vec![1, 2, 3]);
        assert_eq!(point.ndim(), 3);
    }

    #[test]
    fn test_add() {
        let point1 = ZPoint::from_vec(vec![1, 2, 3]);
        let point2 = ZPoint::from_vec(vec![4, 5, 6]);
        let point3 = &point1 + &point2;
        assert_eq!(point3, ZPoint::from_vec(vec![5, 7, 9]));
    }

    #[test]
    fn test_sub() {
        let point1 = ZPoint::from_vec(vec![1, 2, 3]);
        let point2 = ZPoint::from_vec(vec![4, 5, 6]);
        let point3 = &point1 - &point2;
        assert_eq!(point3, ZPoint::from_vec(vec![-3, -3, -3]));
    }

    #[test]
    fn test_mul() {
        let point1 = ZPoint::from_vec(vec![1, 2, 3]);
        let point2 = ZPoint::from_vec(vec![4, 5, 6]);
        let point3 = &point1 * &point2;
        assert_eq!(point3, ZPoint::from_vec(vec![4, 10, 18]));
    }

    #[test]
    fn test_rem() {
        let point1 = ZPoint::from_vec(vec![1, 2, 3]);
        let point2 = ZPoint::from_vec(vec![4, 5, 6]);
        let point3 = &point1 % &point2;
        assert_eq!(point3, ZPoint::from_vec(vec![1, 2, 3]));
    }

    #[test]
    fn test_neg() {
        let point1 = ZPoint::from_vec(vec![1, 2, 3]);
        let point2 = -&point1;
        assert_eq!(point2, ZPoint::from_vec(vec![-1, -2, -3]));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZRange {
    start: ZPoint,
    end: ZPoint,
}

impl ZRange {
    pub fn from_shape(shape: &ZPoint) -> Self {
        ZRange::between(&ZPoint::zeros_like(shape), shape)
    }

    pub fn between(start: &ZPoint, end: &ZPoint) -> Self {
        assert_equal_ndim!(start.coords, end.coords, between, ..);

        assert!(
            start.coords.iter().zip(end.coords.iter()).all(|(s, e)| { s <= e }),
            "start must be less than or equal to end: {:?} <= {:?}",
            start.coords,
            end.coords
        );

        ZRange { start: start.clone(), end: end.clone() }
    }

    pub fn from_start_with_shape(start: &ZPoint, shape: &ZPoint) -> Self {
        let end = start + shape;
        ZRange::between(start, &end)
    }

    pub fn ndim(&self) -> usize {
        return self.start.ndim();
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
    fn test_from_shape() {
        let shape = ZPoint::from_vec(vec![1, 2, 3]);
        let range = ZRange::from_shape(&shape);
        assert_eq!(range.start, ZPoint::from_vec(vec![0, 0, 0]));
        assert_eq!(range.end, shape);
    }

    #[test]
    fn test_between() {
        let start = ZPoint::from_vec(vec![1, 2, 3]);
        let end = ZPoint::from_vec(vec![4, 5, 6]);
        let range = ZRange::between(&start, &end);
        assert_eq!(range.start, start);
        assert_eq!(range.end, end);
    }

    #[test]
    fn test_from_start_with_shape() {
        let start = ZPoint::from_vec(vec![1, 2, 3]);
        let shape = ZPoint::from_vec(vec![4, 5, 6]);
        let range = ZRange::from_start_with_shape(&start, &shape);
        assert_eq!(range.start, start);
        assert_eq!(range.end, ZPoint::from_vec(vec![5, 7, 9]));
    }

    #[test]
    #[should_panic]
    fn test_between_panic() {
        ZRange::between(&ZPoint::zeros(2), &ZPoint::ones(3));
    }

    #[test]
    fn test_ndim() {
        let start = ZPoint::from_vec(vec![1, 2, 3]);
        let end = ZPoint::from_vec(vec![4, 5, 6]);
        let range = ZRange::between(&start, &end);
        assert_eq!(range.ndim(), 3);
    }

    #[test]
    fn test_shape_size() {
        let start = ZPoint::from_vec(vec![1, 2, 3]);
        let end = ZPoint::from_vec(vec![4, 5, 6]);
        let range = ZRange::between(&start, &end);
        assert_eq!(range.shape(), ZPoint::from_vec(vec![3, 3, 3]));
        assert_eq!(range.size(), 27);
    }
}
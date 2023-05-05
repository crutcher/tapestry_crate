pub mod zspace;

#[cfg(test)]
mod tests {
    use crate::zspace::*;

    #[test]
    fn test_from_vec() {
        let point: ZPoint = ZPoint::from_vec(vec![1, 2, 3]);
        assert_eq!(point.ndim(), 3);
    }
}

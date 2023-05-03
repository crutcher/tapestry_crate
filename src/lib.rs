mod zspace;

#[cfg(test)]
mod tests {
    #[test]
    fn test_from_vec() {
        let point = crate::zspace::ZPoint::from_vec(vec![1, 2, 3]);
        assert_eq!(point.ndim(), 3);
    }
}
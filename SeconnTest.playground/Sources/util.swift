import Foundation

public extension Array where Element == Float {
    public func sum() -> Float {
        return self.reduce(0, +)
    }

    public func average() -> Float {
        return self.sum() / Float(self.count)
    }
}

public extension Array where Element == Double {
    public func sum() -> Double {
        return self.reduce(0, +)
    }

    public func average() -> Double {
        return self.sum() / Double(self.count)
    }
}

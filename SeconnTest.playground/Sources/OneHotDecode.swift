import Foundation
import Seconn

public enum OneHotDecoderError: Error {
    case invalidInput(value: [FloatType])

    public var localizedDescription: String {
        switch self {
        case let .invalidInput(value):
            return "Invalid input for one-hot decode: \(value)"
        }
    }
}

public func oneHotDecode(_ a: [FloatType]) throws -> Int {
    for i in a.indices {
        if a[i] == 1.0 {
            return i
        }
    }
    throw OneHotDecoderError.invalidInput(value: a)
}

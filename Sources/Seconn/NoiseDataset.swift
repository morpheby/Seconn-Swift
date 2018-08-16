//
//  NoiseDataset.swift
//  Seconn
//
//  Created by Ilya Mikhaltsou on 11.04.2018.
//

import Foundation

public class NoiseDataset {

    public enum RandomFunction {
        case linear(min: FloatType, max: FloatType)
        case gaussian(mean: FloatType, sigma: FloatType)

        func f() -> FloatType {
            switch self {
            case let .linear(min: min, max: max):
                return min + (max - min) * (FloatType(arc4random_uniform(UInt32.max)) / FloatType(UInt32.max))
            case .gaussian:
                fatalError("Not implemented")
            }
        }
    }

    public enum ValueType {
        case single(value: [FloatType])
        case random(count: Int, RandomFunction)

        func value() -> [FloatType] {
            switch self {
            case let .single(v):
                return v
            case let .random(count: count, f):
                return (0 ..< count) .map { _ in f.f() }
            }
        }
    }

    let inputBatches: [[[FloatType]]]
    let outputBatches: [[[FloatType]]]

    public func batch(index: Int) -> ([[FloatType]], [[FloatType]]) {
        return (inputBatches[index], outputBatches[index])
    }

    public var batchCount: Int {
        return inputBatches.count
    }

    public init(input: ValueType, output: ValueType, count: Int, batchSize: Int) {
        self.inputBatches = NoiseDataset.generate(value: input, count: count, batchSize: batchSize)
        self.outputBatches = NoiseDataset.generate(value: output, count: count, batchSize: batchSize)
    }

    static func generate(value: ValueType, count: Int, batchSize: Int) -> [[[FloatType]]] {
        precondition(count % batchSize == 0, "Count should be divisible by batchSize")
        let countInBatch = count / batchSize

        var batches: [[[FloatType]]] = Array()

        for _ in 0 ..< batchSize {
            var batch: [[FloatType]] = Array()
            batch.reserveCapacity(countInBatch)

            for _ in 0 ..< countInBatch {
                batch.append(value.value())
            }
            batches.append(batch)
        }
        return batches
    }
}

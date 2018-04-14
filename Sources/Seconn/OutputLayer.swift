//
//  OutputLayer.swift
//  Seconn
//
//  Created by Ilya Mikhaltsou on 13.04.2018.
//

import Foundation
import Surge

struct OutputLayer {
    let reductionMatrix: Matrix<FloatType>

    init(inputSize: Int, outputSize: Int) {
        precondition(inputSize % outputSize == 0, "Output layer should be a multiple of its input")
        let count = inputSize / outputSize
        let rate = FloatType(outputSize) / FloatType(inputSize)
        let matrix = (0 ..< outputSize) .map { (i: Int) -> [FloatType] in
            let prefixCount = count * i
            let postfixCount = count * (outputSize - i - 1)

            var array: [FloatType] = Array()
            array.reserveCapacity(inputSize)

            if prefixCount != 0 {
                array.append(contentsOf: Array(repeating: 0.0, count: prefixCount))
            }

            array.append(contentsOf: Array(repeating: rate, count: count))

            if postfixCount != 0 {
                array.append(contentsOf: Array(repeating: 0.0, count: postfixCount))
            }

            return array
        }
        reductionMatrix = Matrix(matrix)
    }
}

extension OutputLayer: Layer {
    var inputSize: Int {
        return reductionMatrix.columns
    }

    var outputSize: Int {
        return reductionMatrix.rows
    }

    func process(input: [FloatType]) -> [FloatType] {
        return (reductionMatrix * Matrix([input])′)[column: 0]
    }
}


extension OutputLayer: LearningLayer {
    mutating func learn(input: [FloatType], output: [FloatType], target: [FloatType], weightRate: FloatType, biasRate: FloatType) -> [FloatType] {

        return ceil((reductionMatrix′ * Matrix([target])′)[column: 0])
    }
}

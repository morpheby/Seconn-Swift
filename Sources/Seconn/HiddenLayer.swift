//
//  HiddenLayer.swift
//  Seconn
//
//  Created by Ilya Mikhaltsou on 13.04.2018.
//

import Foundation
import Surge

/// Initial view of how ECO should apply to Binary step. Note that high bias learning rate is a
/// prerequisite, since it controls the activation threshold. Without appropriate bias learnint
/// rate, the output becomes saturated.
struct HiddenLayer {
    var weights: Matrix<FloatType>
    var biases: [FloatType]
    let activationFunction: ([FloatType]) -> [FloatType] = { input in
        return ceil(clip(input, low: 0.0, high: 1.0))
    }

    init(inputSize: Int, outputSize: Int, weightInitializer: () -> FloatType) {
        let allWeights = (0 ..< inputSize*outputSize) .map { _ in
            weightInitializer()
        }
        self.weights = Matrix(rows: outputSize, columns: inputSize, grid: allWeights)
        self.biases = Array(repeating: 0.0, count: outputSize)
        self.weightCorrectionsInit = Matrix(rows: weights.rows, columns: weights.columns, repeatedValue: 1.0)
    }

    private let weightCorrectionsInit: Matrix<FloatType>
}

extension HiddenLayer: Layer {
    var inputSize: Int {
        return weights.columns
    }

    var outputSize: Int {
        return weights.rows
    }

    func process(input: [FloatType]) -> [FloatType] {
        return activationFunction(mul(weights, Matrix(column: input))[column: 0] .+ biases)
    }
}

extension HiddenLayer: LearningLayer {
    mutating func learn(input: [FloatType], output: [FloatType], target: [FloatType], weightRate: FloatType, biasRate: FloatType) -> [FloatType] {
        //        y1 = act(sum(x1..xN * weights) + biases)
        //
        //        if y0_1 == 0
        //          if y1 == 0
        //            biases[y1] += biasRate
        //            if x1 == 0
        //              weights[x1][y1] += weightRate
        //              x1Corr = 0
        //            else x1 == 1
        //              weights[x1][y1] += weightRate (nop)
        //              x1Corr = 0
        //          else y1 == 1
        //            biases[y1] -= biasRate
        //            if x1 == 0
        //              weights[x1][y1] -= weightRate (nop)
        //              x1Corr = 1
        //            else x1 == 1
        //              weights[x1][y1] -= weightRate
        //              x1Corr = 1
        //        else y0_1 == 1
        //          if y1 == 0
        //            biases[y1] += biasRate
        //            if x1 == 0
        //              weights[x1][y1] += weightRate (nop)
        //              x1Corr = 1
        //            else x1 == 1
        //              weights[x1][y1] += weightRate
        //              x1Corr = 1
        //          else y1 == 1
        //            biases[y1] -= biasRate
        //            if x1 == 0
        //              weights[x1][y1] -= weightRate
        //              x1Corr = 0
        //            else x1 == 1
        //              weights[x1][y1] -= weightRate (nop)
        //              x1Corr = 0
        //
        //        weightCorr = [weightRate]
        //        y1 == 1 -> weightCorr[.][y1] *= -1
        //        y0_1 == y1 ->
        //          x1 == 1 -> weightCorr[x1][y1] = 0
        //          x1Corr = 0
        //        y0_1 != y1 ->
        //          x1 == 0 -> weightCorr[x1][y1] = 0
        //          x1Corr = 1
        //
        //        biasCorr = [biasRate]
        //        y1 == 1 -> biasCorr[y1] *= -1

        var weightCorrections = mul(weightRate, weightCorrectionsInit)
        weightCorrections = elmul(weightCorrections, Matrix(repeatElement(output * -1.5 + 0.5, count: weightCorrections.columns))′)
        let targetNotEqualsOutput = abs(target - output)
        let inputIsOne = input

        weightCorrections = elmul(weightCorrections,
                                  matrixOp({x in clip(x, low: 0.0, high: 1.0)},
                                    Matrix(column: neg(targetNotEqualsOutput) + 1.0) * Matrix(row: neg(inputIsOne) + 1.0) +
                                    Matrix(column: targetNotEqualsOutput) * Matrix(row: inputIsOne)))

        var biasCorrections = Array(repeating: biasRate, count: biases.count)
        biasCorrections = biasCorrections * (output * -2.0 + 1.0)
        biasCorrections = biasCorrections * targetNotEqualsOutput

        let targetInput: [FloatType] =
            activationFunction(
//                sum(weights′ * Matrix(column: targetNotEqualsOutput /* * 2.0 - 1.0 */), axies: .row)[column: 0]
                sum(weights′ * Matrix(column: output), axies: .row)
        )

        weights = weights + weightCorrections
        biases = biases .+ biasCorrections

        return targetInput
    }
}

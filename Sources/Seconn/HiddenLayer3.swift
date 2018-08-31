//
//  HiddenLayer3.swift
//  Seconn
//
//  Created by Ilya Mikhaltsou on 12.07.2018.
//

import Foundation
import Surge

/// Initial view of how ECO should apply to Binary step. Note that high bias learning rate is a
/// prerequisite, since it controls the activation threshold. Without appropriate bias learnint
/// rate, the output becomes saturated.
struct HiddenLayer3 {
    var weights: SliceableMatrix<FloatType>
    var biases: [FloatType]
    let activationFunction: ([FloatType]) -> [FloatType] = { input in
        return ceil(clip(input, low: 0.0, high: 1.0))
    }

    init(inputSize: Int, outputSize: Int, weightInitializer: () -> FloatType) {
        let allWeights = (0 ..< inputSize*outputSize) .map { _ in
            weightInitializer()
        }
        self.weights = SliceableMatrix(rows: outputSize, columns: inputSize, grid: allWeights)
        self.biases = Array(repeating: 0.0, count: outputSize)
        self.weightCorrectionsInit = SliceableMatrix(rows: weights.rowCount, columns: weights.columnCount, repeatedValue: 1.0)
    }

    private let weightCorrectionsInit: SliceableMatrix<FloatType>
}

extension HiddenLayer3: Layer {
    var inputSize: Int {
        return weights.columnCount
    }

    var outputSize: Int {
        return weights.rowCount
    }

    func process(input: [FloatType]) -> [FloatType] {
        return activationFunction(mul(weights, SliceableMatrix(column: input))[column: 0] .+ biases)
    }
}

extension HiddenLayer3: LearningLayer {
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
        weightCorrections = elmul(weightCorrections, SliceableMatrix(repeatElement(output * -2.0 + 1.0, count: weightCorrections.columnCount))′)
        let targetNotEqualsOutput = abs(target .- output)
        let inputIsOne = input

        // Simplify expression for Swift compiler
        let weightFormula1 = SliceableMatrix(column: neg(targetNotEqualsOutput) + 1.0) * SliceableMatrix(row: neg(inputIsOne) + 1.0)
        let weightFormula2 = SliceableMatrix(column: targetNotEqualsOutput) * SliceableMatrix(row: inputIsOne)
        weightCorrections = elmul(weightCorrections, weightFormula1 + weightFormula2)

        var biasCorrections = Array(repeating: biasRate, count: biases.count)
        biasCorrections = biasCorrections .* (output * -2.0 + 1.0)
        biasCorrections = biasCorrections .* targetNotEqualsOutput

        let targetInput: [FloatType] =
            activationFunction(
                sum(weights′ * SliceableMatrix(column: targetNotEqualsOutput * 2.0 - 1.0), axies: .row)
        )

        weights = weights + weightCorrections
        biases = biases .+ biasCorrections

        return targetInput
    }
}

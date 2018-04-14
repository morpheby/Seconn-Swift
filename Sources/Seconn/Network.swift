//
//  Network.swift
//  Seconn
//
//  Created by Ilya Mikhaltsou on 6.04.2018.
//

import Foundation
import Surge

public typealias FloatType = Float

protocol Layer {
    func process(input: [FloatType]) -> [FloatType]

    var inputSize: Int { get }

    var outputSize: Int { get }
}

protocol LearningLayer {
    mutating func learn(input: [FloatType], output: [FloatType], target: [FloatType], weightRate: FloatType, biasRate: FloatType) -> [FloatType]
}

public struct SecoNetworkConfiguration {
    var inputSize: Int
    var outputSize: Int

    var hiddenLayersSizes: [Int]

    var weightInitializer: () -> FloatType

    var learningRateForWeights: FloatType
    var learningRateForBiases: FloatType
}

public class SecoNetwork {

    private let config: SecoNetworkConfiguration
    private var layers: [Layer]

    public init(config: SecoNetworkConfiguration) {
        self.config = config

        self.layers = []
        setup()
    }

    private func setup() {
        var lastLayer: Layer = InputLayer(inputSize: config.inputSize)
        layers.append(lastLayer)

        for layerSize in config.hiddenLayersSizes {
            lastLayer = HiddenLayer(inputSize: lastLayer.outputSize,
                                    outputSize: layerSize, weightInitializer: config.weightInitializer)
            layers.append(lastLayer)
        }

        lastLayer = OutputLayer(inputSize: lastLayer.outputSize, outputSize: config.outputSize)
        layers.append(lastLayer)
    }

    public func process(input: [FloatType]) -> [FloatType] {
        return layers.reduce(input) { (lastOutput, layer) -> [FloatType] in
            layer.process(input: lastOutput)
        }
    }

    public func train(input: [FloatType], output: [FloatType], rateReduction: FloatType, inverse: Bool = false) {
        let hiddenInput: [FloatType]
        var hiddenIOs: [([FloatType], [FloatType])] = []

        hiddenInput = layers.first!.process(input: input)

        let _ = layers.dropFirst().reduce(hiddenInput) { (lastOutput, layer) -> [FloatType] in
            let output = layer.process(input: lastOutput)
            hiddenIOs.append((lastOutput, output))
            return output
        }

        let _ = zip(hiddenIOs, layers.indices.filter { layerIdx in layers[layerIdx] is LearningLayer })
            .reversed()
            .reduce(output) { (lastOutput, arg) in
                let (layerIOs, layerIdx) = arg
                let (layerInput, layerOutput) = layerIOs
                var layerTmp = layers[layerIdx] as! Layer & LearningLayer
                let newOutput = layerTmp.learn(input: layerInput, output: layerOutput, target: lastOutput,
                                               weightRate: (inverse ? -1.0 : 1.0) * config.learningRateForWeights * rateReduction,
                                            biasRate: config.learningRateForBiases * rateReduction)
                layers[layerIdx] = layerTmp
                return newOutput
        }
    }
}

extension SecoNetwork: CustomDebugStringConvertible {
    public var debugDescription: String {
        return self.layers.map { layer in
            switch layer {
            case let l as InputLayer:
                return "InputLayer: {inputSize: \(l.inputSize)}"
            case let l as HiddenLayer:
                let fullWeights = Array(l.weights.joined())
                return "HiddenLayer: {inputSize: \(l.inputSize), outputSize: \(l.outputSize), minWeight: \(min(fullWeights)), maxWeight: \(max(fullWeights)), minBias: \(min(l.biases)), maxBias: \(max(l.biases)), oneRow: \(l.weights[row: 0][0..<min(l.weights.columns, 100)])}"
            case let l as OutputLayer:
                let fullWeights = Array(l.reductionMatrix.joined())
                return "OutputLayer: {inputSize: \(l.inputSize), outputSize: \(l.outputSize), minWeight: \(min(fullWeights)), maxWeight: \(max(fullWeights)), oneRow: \(l.reductionMatrix[row: 0])}"
            default:
                return ""
            }
        } .joined(separator: "\n")
    }
}

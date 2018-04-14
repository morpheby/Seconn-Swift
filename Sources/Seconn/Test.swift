//
//  Test.swift
//  Seconn
//
//  Created by Ilya Mikhaltsou on 7.04.2018.
//

import Foundation
import Surge

public func argmax<T: Collection>(_ collection: T) -> T.Index where T.Element: Comparable {
    var maxIdx = collection.indices.first!
    for idx in collection.indices {
        if collection[maxIdx] < collection[idx] {
            maxIdx = idx
        }
    }
    return maxIdx
}

public class SeconnTest {
    public let dataset: MnistDataset
    public let detrainDataset: NoiseDataset

    public let config: SecoNetworkConfiguration

    public var detrainCoeff: FloatType

    public var neuralNetwork: SecoNetwork

    public var debugLevel: DebugLevel = .none

    public enum DebugLevel: Int, Comparable {
        public static func < (lhs: SeconnTest.DebugLevel, rhs: SeconnTest.DebugLevel) -> Bool {
            return lhs.rawValue < rhs.rawValue
        }

        case none = 0
        case info
        case debug
    }

    public func batchCompute(inputBatch: [[FloatType]]) -> [[FloatType]] {
        let outputs = inputBatch.map { t in self.neuralNetwork.process(input: t) }
        return outputs
    }

    public func performance(output: [Int], target: [Int]) -> Float {
        return mean(zip(output, target) .map { a, b in a == b ? 1.0 : 0.0 })
    }

    public func test(batches: CountableRange<Int>) -> FloatType {
        var totalPerf: FloatType = 0.0

        for batchIdx in batches {
            let (testSet, testLabels) = dataset.testBatch(index: batchIdx)

            let outputs = batchCompute(inputBatch: testSet)
            let perf = performance(output: outputs.map(argmax), target: testLabels.map(argmax))
            totalPerf += perf
        }
        return totalPerf / FloatType(batches.count)
    }

    fileprivate class TrainVars {
        let parent: SeconnTest
        var batchSet: [[FloatType]] = []
        var batchLabels: [[FloatType]] = []

        var batchCount: Int { return parent.dataset.trainBatchCount }
        var samplesInBatch: Int { return batchSet.count }

        init(parent: SeconnTest) {
            self.parent = parent
        }

        func fetchBatch(batchIdx: Int) {
            (batchSet, batchLabels) = parent.dataset.trainBatch(index: batchIdx)
        }
    }

    public struct TrainSequence: TrainSequenceProtocol {
        private let parent: SeconnTest

        public func makeIterator() -> Iterator {
            return Iterator(parent: self.parent)
        }

        fileprivate init(parent: SeconnTest) {
            self.parent = parent
        }

        public struct TrainState: TrainStateProtocol {
            public let epoch: Int
            public let batch: Int
            public let sample: Int
            public let current: Int

            private let vars: TrainVars

            fileprivate init(epoch: Int, batch: Int, sample: Int, current: Int, vars: TrainVars) {
                self.epoch = epoch
                self.batch = batch
                self.sample = sample
                self.current = current
                self.vars = vars
            }

            public func train(rate: FloatType) {
                vars.parent.neuralNetwork.train(input: vars.batchSet[sample], output: vars.batchLabels[sample], rateReduction: rate)
            }
        }

        public struct Iterator: TrainIteratorProtocol {
            public var epochIdx: Int = 0
            public var batchIdx: Int = 0
            public var sampleIdx: Int = 0
            public var batchCount: Int { return vars.batchCount }
            public var samplesInBatchCount: Int { return vars.samplesInBatch }

            private var vars: TrainVars

            public typealias Element = TrainState

            private mutating func fetchBatch() {
                if isKnownUniquelyReferenced(&vars) {
                    vars.fetchBatch(batchIdx: self.batchIdx)
                } else {
                    self.vars = TrainVars(parent: self.vars.parent)
                    vars.fetchBatch(batchIdx: self.batchIdx)
                }
            }

            public mutating func next() -> Element? {
                sampleIdx += 1
                if sampleIdx >= samplesInBatchCount {
                    batchIdx += 1
                    sampleIdx = 0

                    if batchIdx >= batchCount {
                        epochIdx += 1
                        batchIdx = 0
                        self.reportEpoch()
                    }
                    fetchBatch()
                }

                return TrainState(epoch: epochIdx, batch: batchIdx, sample: sampleIdx, current: currentIdx, vars: vars)
            }

            private func reportEpoch() {
                if self.vars.parent.debugLevel >= .info {
                    print("Epoch \(epochIdx)")
                }
            }

            fileprivate init(parent: SeconnTest) {
                self.vars = TrainVars(parent: parent)
                self.fetchBatch()
                self.reportEpoch()
            }
        }
    }

    public struct CountLimitedTrainSequence<T: TrainSequenceProtocol>: TrainSequenceProtocol {

        public struct Iterator: TrainIteratorProtocol {
            public var epochIdx: Int { return internalTrain.epochIdx }
            public var batchIdx: Int { return internalTrain.batchIdx }
            public var sampleIdx: Int { return internalTrain.sampleIdx }
            public var batchCount: Int { return internalTrain.batchCount }
            public var samplesInBatchCount: Int { return internalTrain.samplesInBatchCount }

            public typealias Element = TrainState

            fileprivate var internalTrain: T.Iterator
            let limitedCount: Int

            public mutating func next() -> Element? {
                if currentIdx < limitedCount {
                    guard let s = internalTrain.next() else { return nil }
                    return TrainState(trainState: s, limitedCount: limitedCount)
                } else {
                    return nil
                }
            }

            fileprivate init(internalTrain: T.Iterator, limitedCount: Int) {
                self.internalTrain = internalTrain
                self.limitedCount = limitedCount
            }
        }

        public struct TrainState: LimitedTrainStateProtocol {
            public var epoch: Int { return trainState.epoch }
            public var batch: Int { return trainState.batch }
            public var sample: Int { return trainState.sample }
            public var current: Int { return trainState.current }

            public let limitedCount: Int

            public func train(rate: FloatType) {
                trainState.train(rate: rate)
            }

            private let trainState: TrainStateProtocol

            fileprivate init(trainState: TrainStateProtocol, limitedCount: Int) {
                self.trainState = trainState
                self.limitedCount = limitedCount
            }
        }

        let limitedCount: Int
        var internalTrain: T

        public func makeIterator() -> Iterator {
            return Iterator(internalTrain: self.internalTrain.makeIterator(), limitedCount: limitedCount)
        }
    }

    public func train() -> TrainSequence {
        return TrainSequence(parent: self)
    }

    public init() throws {
        dataset = try loadMnistDataset()

        detrainDataset = NoiseDataset(
            input: .random(count: 784, .linear(min: 0.0, max: 1.0)),
            output: .single(value: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            count: 100 * dataset.trainBatchCount,
            batchSize: 100)

        detrainCoeff = 0.1

        config = SecoNetworkConfiguration(
            inputSize: 784,
            outputSize: 10,
//            hiddenLayersSizes: [100, 50],
            hiddenLayersSizes: [1000],
//            hiddenLayersSizes: [1000, 1000],
            weightInitializer: { () -> FloatType in
                Float(arc4random_uniform(UInt32.max)) / Float(UInt32.max) - 0.5
            },
            learningRateForWeights: 0.001,
//            learningRateForBiases: 0.0
            learningRateForBiases: 0.01
        )

        neuralNetwork = SecoNetwork(config: config)
    }
}

public protocol TrainStateProtocol {
    var epoch: Int { get }
    var batch: Int { get }
    var sample: Int { get }
    var current: Int { get }

    func train(rate: FloatType)
}

public protocol TrainSequenceProtocol: Sequence where Iterator: TrainIteratorProtocol {
}

public protocol TrainIteratorProtocol: IteratorProtocol where Element: TrainStateProtocol {
    var epochIdx: Int { get }
    var batchIdx: Int { get }
    var sampleIdx: Int { get }
    var batchCount: Int { get }
    var samplesInBatchCount: Int { get }
}

public protocol LimitedTrainStateProtocol: TrainStateProtocol {
    var limitedCount: Int { get }
    var percentCompleted: Float { get }
}

extension TrainIteratorProtocol {
    fileprivate var currentIdx: Int {
        return epochIdx * self.batchCount +
            batchIdx * self.samplesInBatchCount +
            sampleIdx
    }
}

extension LimitedTrainStateProtocol {
    public var percentCompleted: Float {
        return Float(current * 100) / Float(self.limitedCount)
    }

    public func printStatus() {
        if (current * 100 % limitedCount) == 0 {
            print("\n\(percentCompleted)%", terminator: " ")
        } else {
            print(".", terminator: "")
        }

        // Fix debug output done by line
        if (current % 10) == 0 {
            fflush(stdout)
        }
    }
}

extension TrainSequenceProtocol {
    public func prefix(_ maxLength: Int) -> SeconnTest.CountLimitedTrainSequence<Self> {
        return SeconnTest.CountLimitedTrainSequence(limitedCount: maxLength, internalTrain: self)
    }
}


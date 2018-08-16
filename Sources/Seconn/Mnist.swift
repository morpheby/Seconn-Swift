//
//  Mnist.swift
//  seconn
//
//  Created by Ilya Mikhaltsou on 6.04.2018.
//

import Foundation
import NnMnist

public protocol MnistDatasetProtocol {
    var trainBatchCount: Int { get }
    var testBatchCount: Int { get }

    func trainBatch(index: Int) -> ([[Float]], [[Float]])
    func testBatch(index: Int) -> ([[Float]], [[Float]])
}

public struct MnistDataset: MnistDatasetProtocol {
    private let datasetManager: MNISTManager

    init(datasetManager: MNISTManager) {
        self.datasetManager = datasetManager
        print("Loaded")
    }

    public var trainBatchCount: Int {
        return datasetManager.trainImages.count
    }

    public var testBatchCount: Int {
        return datasetManager.validationImages.count
    }

    public func trainBatch(index: Int) -> ([[Float]], [[Float]]) {
        return (
            datasetManager.trainImages[index],
            datasetManager.trainLabels[index]
        )
    }

    public func testBatch(index: Int) -> ([[Float]], [[Float]]) {
        return (
            datasetManager.validationImages[index],
            datasetManager.validationLabels[index]
        )
    }
}

public struct MnistDatasetStratified: MnistDatasetProtocol {
    private let trainBatches: ([[[Float]]], [[[Float]]])
    private let testBatches: [([[Float]], [[Float]])]

    init(internalDataset: MnistDatasetProtocol) {
        let trainBatchCount = internalDataset.trainBatchCount
        let testBatchCount = internalDataset.testBatchCount

        let fullBatch: ([[Float]], [[Float]]) = (0..<trainBatchCount).map { (i: Int) -> ([[Float]], [[Float]]) in
            return internalDataset.trainBatch(index: i)
        } .reduce(into: ([], [])) { (result, value) in
            result.0.append(contentsOf: value.0)
            result.1.append(contentsOf: value.1)
        }
        var stratas: [([[Float]], [[Float]])] = (0..<10).map { i in
            let indices = fullBatch.1.indices.filter { fullBatch.1[$0][i] == 1.0 }
            return (
                indices.map { j in fullBatch.0[j] },
                indices.map { j in fullBatch.1[j] }
            )
        }
        let minSize = stratas.map { $0.0.count } .min()!
        let strataBatchSize = minSize / trainBatchCount
        let strataFullSize = strataBatchSize * trainBatchCount
        for i in stratas.indices {
            stratas[i].0.removeLast(stratas[i].0.count - strataFullSize)
            stratas[i].1.removeLast(stratas[i].1.count - strataFullSize)
        }

        let batches: ([[[Float]]], [[[Float]]]) = stratas.map { a in
            (0..<trainBatchCount).map { (i: Int) -> (ArraySlice<[Float]>, ArraySlice<[Float]>) in
                let start = i * strataBatchSize
                let end = (i + 1) * strataBatchSize
                return (a.0[start..<end], a.1[start..<end])
            }
        } .reduce(into: (Array<[[Float]]>(repeating: [], count: trainBatchCount), Array<[[Float]]>(repeating: [], count: trainBatchCount))) { (result, value) in
            for i in result.0.indices {
                result.0[i].append(contentsOf: value[i].0)
                result.1[i].append(contentsOf: value[i].1)
            }
        }

        self.trainBatches = batches
        self.testBatches = (0..<testBatchCount).map { i in internalDataset.testBatch(index: i) }

        self.trainBatchCount = trainBatchCount
        self.testBatchCount = testBatchCount
        
        print("Stratified")
        print("Strata batch size: \(strataBatchSize)")
    }

    public let trainBatchCount: Int

    public let testBatchCount: Int

    public func trainBatch(index: Int) -> ([[Float]], [[Float]]) {
        return (trainBatches.0[index], trainBatches.1[index])
    }

    public func testBatch(index: Int) -> ([[Float]], [[Float]]) {
        return testBatches[index]
    }
}

public func loadMnistDataset() throws -> MnistDataset {
    let mnistDir = FileManager.default.temporaryDirectory.appendingPathComponent("MNIST")
    try? FileManager.default.createDirectory(at: mnistDir, withIntermediateDirectories: false, attributes: nil)
    
    let m: MNISTManager = try MNISTManager(
        directory: mnistDir,
        pixelRange: (0.0, 1.0), batchSize: 100,
        shuffle: false
    )
    return MnistDataset(datasetManager: m)
}


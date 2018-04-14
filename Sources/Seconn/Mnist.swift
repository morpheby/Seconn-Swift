//
//  Mnist.swift
//  seconn
//
//  Created by Ilya Mikhaltsou on 6.04.2018.
//

import Foundation
import NnMnist

public struct MnistDataset {
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


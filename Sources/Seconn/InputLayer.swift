//
//  InputLayer.swift
//  Seconn
//
//  Created by Ilya Mikhaltsou on 13.04.2018.
//

import Foundation
import Surge

struct InputLayer {
    let inputSize: Int
    let activationFunction: ([FloatType]) -> [FloatType] = { input in
        return ceil(clip(input, low: 0.0, high: 1.0))
    }
}

extension InputLayer: Layer {
    var outputSize: Int {
        return inputSize
    }

    func process(input: [FloatType]) -> [FloatType] {
        return activationFunction(input)
    }
}

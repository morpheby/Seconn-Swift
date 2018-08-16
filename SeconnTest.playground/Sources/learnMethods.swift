import Foundation
import Seconn

let repeats = 5
let learnCutoff: FloatType = 1.0
let delearnCutoff: FloatType = 0.5
let delearnRate: FloatType = 0.25

public func doLearn(_ test: SeconnTest, state: TrainStateProtocol, rate: FloatType) {
    for _ in 0..<repeats {
        let output = state.process()
        let outputIdx: Int = argmax(output)
        let target = state.value.output
        let targetIdx = try! oneHotDecode(target)

        print("\(targetIdx), \(outputIdx), " + output.map(String.init).joined(separator: ", "), terminator: ", ")

        state.train(rate: rate)

        do {
            let output = state.process()
            let outputIdx: Int = argmax(output)
            let target = state.value.output
            let targetIdx = try! oneHotDecode(target)
            print("\(outputIdx), " + output.map(String.init).joined(separator: ", "))
        }
    }
}

public func doLearnMagic(_ test: SeconnTest, state: TrainStateProtocol, rate: FloatType) {
    for _ in 0..<repeats {
        let output = state.process()
        let outputIdx: Int = argmax(output)
        let target = state.value.output
        let targetIdx = try! oneHotDecode(target)

        print("\(targetIdx), \(outputIdx), " + output.map(String.init).joined(separator: ", "), terminator: ", ")

        if outputIdx != targetIdx {
            let delearnTargets = output.enumerated().filter({ i,e in e > delearnCutoff && i != targetIdx })
            if delearnTargets.count != 0 {
                var delearnVector: [FloatType] = Array(repeating: 0, count: 10)
                for (i,_) in delearnTargets { delearnVector[i] = 1.0 }
                test.neuralNetwork.train(input: state.value.input, output: delearnVector, rateReduction: rate*delearnRate, inverse: true)
            }
        }

        if output[targetIdx] <= learnCutoff {
            state.train(rate: rate)
        }

        do {
            let output = state.process()
            let outputIdx: Int = argmax(output)
            let target = state.value.output
            let targetIdx = try! oneHotDecode(target)
            print("\(outputIdx), " + output.map(String.init).joined(separator: ", "))
        }
    }
}


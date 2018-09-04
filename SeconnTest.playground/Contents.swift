
import Cocoa
import Surge
import Seconn

/*:
 This playground shows using of S-ECO network with Binary step activation
 function.

 For general reference on ECO network, see [ECO Overview](https://blog.morphe.by/error-curvature-optimization-for-neural-networks/) on my blog.
 */



do {
 let test: SeconnTest = try SeconnTest()
 test.debugLevel = .info

 let rateDefault: Float = 0.8
 var rate: Float = rateDefault
 var rateDecay: Float = 1.000
 var skipper = 0
 let skipValue = 10

 print("Rate: \(rateDefault)")
 print("Rate decay: \(rateDecay)")

 func doTest(_ test: SeconnTest) {
     if skipper == 0 {
         skipper = skipValue
         let t = test.test(batches: 0..<3)
         t.totalPerformance
         t.indexedPerformance
     } else {
         skipper -= 1
         let t = test.test(batches: 0..<1)
         t.totalPerformance
         t.indexedPerformance
     }
 }

 if false {
     let targetPerf: FloatType = 0.8
     let attempts = 15
     var counts = Array(repeating: 0, count: 10)
     var states = (0..<10).map { label in
         test.train().filtered { e in (try! oneHotDecode(e.value.output)) == label }
             .makeIterator()
     }
     print("Using indexed method with target=\(targetPerf), attempts=\(attempts)")
     while let labelValue = (test.test(batches: 0..<1).indexedPerformance
             .drop { $0 >= targetPerf } .indices.first) {
         print("Label: \(labelValue)")
         for _ in 0..<attempts {
             guard let next = states[labelValue].next() else { break }
             doLearn(test, state: next, rate: rate)
             rate *= 1.000
             doTest(test)
         }
         counts[labelValue] += attempts
         counts
     }
 } else if false {
 //    let seq = [9,0,5,6,3,8,2,1,7,4]
     let seq = [0,1,4,8,9,3,6,5,7]
 //    let seq = [0,1,4,8,0,8,1,4,2,8,0,2,9,0,4,8,1,3,6,3,5,0,6,8,9,8,5,7,4,1,2,6,7,8]
     print("Using sequence method with sequence=\(seq)")
     for labelValue in seq {
         print("Label: \(labelValue)")
         rate = rateDefault
         var count = 0
         for trainState in (test.train().filtered { e in (try! oneHotDecode(e.value.output)) == labelValue }.prefix(450)) {
             doLearn(test, state: trainState, rate: rate)
             rate *= rateDecay
             count += 1
             doTest(test)
         }
         print("Label count: \(count)")
     }
 } else {
     print("Using plain method")
     for trainState in test.train().prefix(500).dropFirst(250) {
         doLearn(test, state: trainState, rate: rate)
         rate *= rateDecay
         doTest(test)
     }
 }

 rate = 0.5
 rateDecay = 0.995

 print("Rate: \(rateDefault)")
 print("Rate decay: \(rateDecay)")

 if false {
     print("Using plain method")
     for trainState in test.train().prefix(250) {
         doLearn(test, state: trainState, rate: rate)
         rate *= rateDecay
         doTest(test)
     }
 }

 test.neuralNetwork
 print(test.neuralNetwork)

 /*:
  Internally, this network is set to use 1000 neurons. Obviously,
  you can't get high performance with binary step and such a little quantity
  of neurons on MNIST dataset. But it also shows that it works, and doesn't
  suffer from EC-saturation and value locking, which happen when using
  Laplacian operator on common networks.
  */
 let testResult = test.test(batches: 0..<10)

 let trainPerformance = test.train().prefix(250).map { s in argmax(s.process()) == (try! oneHotDecode(s.value.output)) ? 1.0 : 0.0  }.average()
 print("Train performance: \(trainPerformance)")
 print("Test performance: \(testResult.totalPerformance)")
 print("Test performance (indexed): \(testResult.indexedPerformance)")

 let target = test.dataset.testBatch(index: 0).1.map(argmax)
 let result = test.batchCompute(inputBatch: test.dataset.testBatch(index: 0).0)

 print(zip(target, result).map { t,r in "\(t): \(argmax(r)) \(r)"}, separator: "\n")
}
catch let error {
    print(error.localizedDescription)
    abort()
}




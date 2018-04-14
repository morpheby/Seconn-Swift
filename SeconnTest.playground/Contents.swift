
import Cocoa
import Surge
import Seconn

/*:
 This playground shows using of S-ECO network with Binary step activation
 function.

 For general reference on ECO network, see [ECO Overview](https://blog.morphe.by/error-curvature-optimization-for-neural-networks/) on my blog.
 */

let test: SeconnTest

do {
    test = try SeconnTest()
}
catch let error {
    print(error.localizedDescription)
    abort()
}

test.debugLevel = .info

var rate: Float = 3.0

for trainState in test.train().prefix(250) {
    trainState.train(rate: rate)
    rate *= 0.995
/*:
 Observe here for performance values:
 */
    test.test(batches: 0..<1)
}

rate = 0.5

for trainState in test.train().prefix(250) {
    trainState.train(rate: 0.5)
    rate *= 0.995
/*:
 And here too:
 */
    test.test(batches: 0..<1)
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
test.test(batches: 0..<10)

let target = test.dataset.testBatch(index: 0).1.map(argmax)
let result = test.batchCompute(inputBatch: test.dataset.testBatch(index: 0).0)

print(zip(target, result).map { t,r in "\(t): \(argmax(r)) \(r)"}, separator: "\n")


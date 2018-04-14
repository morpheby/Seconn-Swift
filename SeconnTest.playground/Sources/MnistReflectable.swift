import Foundation
import Seconn

// Silence MnistDataset playground representation. Otherwise, playground chokes on class members
extension MnistDataset: CustomReflectable {
    public var customMirror: Mirror {
        return Mirror(self, children: [
            (label: "trainBatchCount", value: self.trainBatchCount),
            (label: "testBatchCount", value: self.testBatchCount),
        ])
    }
}


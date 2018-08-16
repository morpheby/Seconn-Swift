import Foundation
import Seconn

extension SeconnTest: CustomReflectable {
    public var customMirror: Mirror {
        let children: [Mirror.Child] = [
            (label: "config", value: self.config),
            (label: "dataset", value: self.dataset),
            (label: "neuralNetwork", value: self.neuralNetwork),
        ]
        return Mirror(self, children: children, displayStyle: .`struct`)
    }
}

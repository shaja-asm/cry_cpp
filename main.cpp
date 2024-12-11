#include <iostream>
#include <chrono>
#include <thread>
#include "CryDetector.h"
#include "CryAnnotation.h"

int main() {
    CryDetector detector;
    detector.start();

    // Example main loop
    for (int i = 0; i < 3; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(10));
        CryAnnotation state = detector.get_cry_state();
        std::cout << "Current cry state: " << (int)state << std::endl;
    }

    detector.stop();
    return 0;
}

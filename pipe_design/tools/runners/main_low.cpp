#include "formats/SearchInstance.h"
#include "runners/LowLevelRunner.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string instance_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--instance" || arg == "-i") && i + 1 < argc) {
            instance_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: pipe_run_low --instance <file.json>\n"
                << "  Writes under SearchInstance.output_dir when set:\n"
                << "    reports.jsonl\n"
                << "    closed_set/closed_q*.json   if export_closed_set\n"
                << "    posthoc/trace_q*.json       if export_posthoc\n";
            return 0;
        } else {
            std::cerr << "Unknown arg: " << arg << '\n';
            return 1;
        }
    }

    if (instance_path.empty()) {
        std::cerr << "pipe_run_low: --instance is required\n";
        return 1;
    }

    try {
        const auto instance = pipe_design::tools::SearchInstance::from_file(instance_path);
        pipe_design::tools::LowLevelRunner{}.run(instance);
    } catch (const std::exception& e) {
        std::cerr << "pipe_run_low: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

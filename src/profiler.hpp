#pragma once
#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <mutex>
#include <iomanip>

namespace sankhya {
namespace profile {

struct TimerRecord {
    double total_ms = 0.0;
    int count = 0;
};

inline std::map<std::string, TimerRecord>& get_records() {
    static std::map<std::string, TimerRecord> records;
    return records;
}

inline std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>>& get_cpu_starts() {
    static std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>> starts;
    return starts;
}

inline std::mutex& get_mutex() {
    static std::mutex m;
    return m;
}

inline void add_time(const std::string& name, double ms) {
    std::lock_guard<std::mutex> lock(get_mutex());
    get_records()[name].total_ms += ms;
    get_records()[name].count++;
}

inline void start_cpu(const std::string& name) {
    std::lock_guard<std::mutex> lock(get_mutex());
    get_cpu_starts()[name] = std::chrono::steady_clock::now();
}

inline void stop_cpu(const std::string& name) {
    auto end = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(get_mutex());
    auto it = get_cpu_starts().find(name);
    if (it != get_cpu_starts().end()) {
        double ms = std::chrono::duration<double, std::milli>(end - it->second).count();
        get_records()[name].total_ms += ms;
        get_records()[name].count++;
    }
}

inline void print_report(const std::string& instance, int iterations, const std::string& status) {
    std::lock_guard<std::mutex> lock(get_mutex());
    auto& recs = get_records();
    
    std::cerr << "\n============================================================\n";
    std::cerr << "SANKHYA PROFILING REPORT\n";
    std::cerr << "============================================================\n\n";
    std::cerr << "Instance: " << instance << "\n\n";
    
    auto print_line = [&](const std::string& label, const std::string& key) {
        double ms = 0.0;
        if (recs.count(key)) ms = recs[key].total_ms;
        std::cerr << std::left << std::setw(24) << label << ": " 
                  << std::right << std::setw(10) << std::fixed << std::setprecision(3) << ms << " ms\n";
    };

    print_line("Parsing", "Parsing");
    print_line("Model construction", "Model construction");
    print_line("Presolve", "Presolve");
    print_line("GPU initialization", "GPU initialization");
    print_line("GPU allocation", "GPU allocation");
    print_line("H->D transfer", "H->D transfer");
    print_line("Solver setup", "Solver setup");
    print_line("Simplex", "Simplex");
    print_line("  Pivot selection", "Pivot selection");
    print_line("  Basis update", "Basis update");
    print_line("  GPU kernels", "GPU kernels");
    print_line("  Synchronization", "Synchronization");
    print_line("Fallback", "Fallback");
    print_line("D->H transfer", "D->H transfer");
    print_line("Certificate", "Certificate");
    print_line("KKT verification", "KKT verification");
    print_line("Cleanup", "Cleanup");
    
    double total_ms = 0.0;
    for (auto& p : recs) {
        if (p.first != "TOTAL" && p.first != "Pivot selection" && p.first != "Basis update" && p.first != "GPU kernels" && p.first != "Synchronization") {
            // total approx sum of disjoint components
        }
    }
    
    if (recs.count("TOTAL")) {
        std::cerr << "\nTOTAL:                   " << std::fixed << std::setprecision(3) << recs["TOTAL"].total_ms << " ms\n";
    }
    std::cerr << "Iterations:              " << iterations << "\n";
    std::cerr << "Status:                  " << status << "\n\n";
    
    // Detailed fallback info
    int fallback_count = recs.count("Fallback") ? recs["Fallback"].count : 0;
    double fallback_time = recs.count("Fallback") ? recs["Fallback"].total_ms : 0.0;
    std::cerr << "Fallback triggered: " << (fallback_count > 0 ? "YES" : "NO") << "\n";
    std::cerr << "Fallback invocation count: " << fallback_count << "\n";
    std::cerr << "Total fallback time: " << std::fixed << std::setprecision(3) << fallback_time << " ms\n";
    if (fallback_count > 0) {
        std::cerr << "Average fallback time: " << std::fixed << std::setprecision(3) << fallback_time / fallback_count << " ms\n";
    }
    
    // KKT info
    int kkt_count = recs.count("KKT verification") ? recs["KKT verification"].count : 0;
    double kkt_time = recs.count("KKT verification") ? recs["KKT verification"].total_ms : 0.0;
    std::cerr << "\nKKT verification executed: " << (kkt_count > 0 ? "YES" : "NO") << "\n";
    std::cerr << "KKT verification time: " << std::fixed << std::setprecision(3) << kkt_time << " ms\n";
    
    // Table
    std::cerr << "\nComponent                         Time        % Total\n";
    std::cerr << "------------------------------------------------------\n";
    
    auto print_pct = [&](const std::string& label, const std::string& key) {
        double ms = recs.count(key) ? recs[key].total_ms : 0.0;
        double pct = (recs.count("TOTAL") && recs["TOTAL"].total_ms > 0) ? (ms / recs["TOTAL"].total_ms * 100.0) : 0.0;
        std::cerr << std::left << std::setw(30) << label << std::right << std::setw(8) << std::fixed << std::setprecision(3) << ms << " ms   " << std::setw(6) << std::setprecision(2) << pct << "%\n";
    };
    
    print_pct("Parsing", "Parsing");
    print_pct("Model construction", "Model construction");
    print_pct("Presolve", "Presolve");
    print_pct("GPU initialization", "GPU initialization");
    print_pct("H->D transfer", "H->D transfer");
    print_pct("Solver setup", "Solver setup");
    print_pct("Simplex", "Simplex");
    print_pct("Fallback", "Fallback");
    print_pct("D->H transfer", "D->H transfer");
    print_pct("Certificate", "Certificate");
    print_pct("KKT verification", "KKT verification");
    print_pct("Cleanup", "Cleanup");
    
    std::cerr << "------------------------------------------------------\n";
}

} // profile
} // sankhya

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

    std::cerr << "\n============================================================\n";
    std::cerr << "SOLVER SETUP DEEP PROFILE\n";
    std::cerr << "=========================\n\n";
    std::cerr << "Instance: " << instance << "\n\n";

    print_line("Solver setup total", "Solver setup");
    std::cerr << "\n";
    print_line("CPU model preparation", "CPU model preparation");
    print_line("Matrix conversion", "Matrix conversion");
    print_line("CSR construction", "CSR construction");
    print_line("CSC construction", "CSC construction");
    print_line("Basis initialization", "Basis initialization");
    print_line("Basis factorization setup", "Basis factorization setup");
    print_line("CUDA library initialization", "CUDA library initialization");
    print_line("CUDA stream setup", "CUDA stream setup");
    print_line("GPU data structure setup", "GPU data structure setup");
    print_line("GPU buffer preparation", "GPU buffer preparation");
    print_line("Device initialization", "Device initialization");
    print_line("Setup synchronization", "Setup synchronization");
    print_line("Other setup", "Other setup");
    std::cerr << "\n";

    std::cerr << "\n============================================================\n";
    std::cerr << "GPU KERNEL PROFILE\n";
    std::cerr << "==================\n\n";
    std::cerr << "Instance: " << instance << "\n\n";
    std::cerr << std::left << std::setw(30) << "Kernel" << std::right << std::setw(10) << "Calls" << std::setw(15) << "Total ms" << std::setw(15) << "Avg ms\n";
    std::cerr << "----------------------------------------------------------------------\n";

    double total_gpu = 0.0;
    for (const auto& p : recs) {
        if (p.first.find("kernel_") == 0) {
            double avg = p.second.total_ms / p.second.count;
            std::cerr << std::left << std::setw(30) << p.first << std::right << std::setw(10) << p.second.count 
                      << std::setw(15) << std::fixed << std::setprecision(3) << p.second.total_ms 
                      << std::setw(15) << std::fixed << std::setprecision(3) << avg << "\n";
            total_gpu += p.second.total_ms;
        }
    }
    std::cerr << "\nTotal GPU kernel time: " << std::fixed << std::setprecision(3) << total_gpu << " ms\n";

    std::cerr << "\n============================================================\n";
    std::cerr << "KKT VERIFICATION DEEP PROFILE\n";
    std::cerr << "=============================\n\n";
    print_line("KKT verification total", "KKT verification");
    std::cerr << "\n";
    print_line("Primal check", "KKT primal check");
    print_line("Dual check", "KKT dual check");
    print_line("Comp slackness check", "KKT comp slackness check");
    print_line("Vector/matrix ops", "KKT vector/matrix operations");
    print_line("Sync and reduction", "KKT synchronization and reduction");
    print_line("Residual calculation", "KKT residual calculation");

    std::cerr << "\n============================================================\n";
    std::cerr << "CUDA SYNCHRONIZATION PROFILE\n";
    std::cerr << "============================\n\n";

    double total_sync = 0.0;
    for (const auto& p : recs) {
        if (p.first.find("Sync: ") == 0) {
            print_line(p.first, p.first);
            total_sync += p.second.total_ms;
        }
    }
    std::cerr << "\nTotal synchronization time: " << std::fixed << std::setprecision(3) << total_sync << " ms\n";

    
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

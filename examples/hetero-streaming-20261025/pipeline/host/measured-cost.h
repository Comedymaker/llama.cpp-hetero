#pragma once
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
struct measured_tree_curve {
    struct point { int rows; double ms; };
    std::vector<point> points;
    double offset_ms = 0;
    bool load(const char *path) {
        std::ifstream f(path); int rows; double ms;
        while(f >> rows >> ms) {
            if(rows<1 || !std::isfinite(ms) || ms<=0 || (!points.empty() && rows<=points.back().rows)) return false;
            // Upper monotone envelope avoids rewarding noise-induced negative marginal cost.
            points.push_back({rows,points.empty()?ms:std::max(ms,points.back().ms)});
        }
        return f.eof() && points.size()>=2;
    }
    double raw(int rows) const {
        if(rows<=points.front().rows) return points.front().ms;
        for(size_t i=1;i<points.size();++i) if(rows<=points[i].rows) {
            const auto &a=points[i-1];const auto &b=points[i];
            return a.ms+(b.ms-a.ms)*double(rows-a.rows)/(b.rows-a.rows);
        }
        return points.back().ms;
    }
    double estimate(int rows) const { return std::max(1.0,raw(rows)+offset_ms); }
    void observe(int rows,double ms) { offset_ms=.8*offset_ms+.2*(ms-raw(rows)); }
};

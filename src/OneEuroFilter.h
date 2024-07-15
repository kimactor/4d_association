#pragma once
#include<iostream>

namespace ZAJX
{
    namespace CaveAlgo
    {

#ifndef PI
#define PI  3.14159265358979323846
#define PI_FL  3.141592f
#endif
        class LowPassFilter
        {
        public:
            LowPassFilter()
            {
                m_initialized = false;
                m_x_previous = 0;
            }
            
            double filter(double x, double alpha = 0.5)
            {
                if (!m_initialized)
                {
                    m_x_previous = x;
                    m_initialized = true;
                    return x;
                }
                double x_filtered = alpha * x + (1 - alpha) * m_x_previous;
                m_x_previous = x_filtered;
                return x_filtered;
            }
        private:
            bool m_initialized;
            double m_x_previous;
        };


        class OneEuroFilter
        { 
        public:
            OneEuroFilter(double freq = 100, double mincutoff = 1, double beta = 0.05, double dcutoff = 1.0)
            {
                m_freq = freq;
                m_mincutoff = mincutoff;
                m_beta = beta;
                m_dcutoff = dcutoff;
                m_dx = 0;
                m_initialized = false;
                m_x_previous = DBL_MAX;
            }

            double getAlpha(double cutoff = 1)
            {
                double tau = 1.0 / (2 * PI *cutoff);
                double te = 1.0 / m_freq;
                return 1.0 / (1.0 + tau / te);
            }

            double filter(double x)
            {
                if (!m_initialized)
                {
                    m_dx = 0;
                    m_initialized = true;
                }
                else {
                    m_dx = (x - m_x_previous) * m_freq;
                }
                double dx_smoothed = filter_dx.filter(m_dx, getAlpha(m_dcutoff));
                double cutoff = m_mincutoff + m_beta * std::fabs(dx_smoothed);
                double x_filtered = filter_x.filter(x, getAlpha(cutoff));
                m_x_previous = x;
                return x_filtered;
            }


        private:
            double m_freq;
            double m_mincutoff;
            double m_beta;
            double m_dcutoff;
            double m_dx;
            bool m_initialized;
            double m_x_previous;
            LowPassFilter filter_x;
            LowPassFilter filter_dx;
        };
    }
}

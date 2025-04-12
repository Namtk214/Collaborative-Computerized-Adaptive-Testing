import React, { useState, useEffect } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

const NormalDistributionChart = ({ theta }) => {
  const [data, setData] = useState([]);
  const [cdfValue, setCdfValue] = useState(0.5);

  useEffect(() => {
    const generateData = () => {
      const newData = [];
      for (let x = -3.5; x <= 3.5; x += 0.05) {
        const roundedX = parseFloat(x.toFixed(2));
        const yValue = normalPDF(x);
        newData.push({
          x: roundedX,
          y: yValue,
          yArea: roundedX <= theta ? yValue : 0
        });
      }
      return newData;
    };

    setData(generateData());
    setCdfValue(standardNormalCDF(theta));
  }, [theta]);

  const normalPDF = (x) => (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-(x * x) / 2);

  const standardNormalCDF = (x) => {
    const erf = (z) => {
      const t = 1.0 / (1.0 + 0.3275911 * Math.abs(z));
      const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429;
      const sign = z < 0 ? -1 : 1;
      const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-z * z);
      return sign * y;
    };
    return 0.5 * (1 + erf(x / Math.sqrt(2)));
  };

  const formatPercentage = (val) => (val * 100).toFixed(2) + '%';

  return (
    <div style={{ padding: '1rem', background: '#fff', borderRadius: '8px', marginTop: '20px' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '1rem' }}>Normal Distribution Visualization</h2>
      <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
        <span style={{ fontWeight: 'bold', color: 'orange' }}>
          Theta (z-score): {theta.toFixed(2)}
        </span>
      </div>
      <div style={{ marginBottom: '1rem', textAlign: 'center' }}>
        <span style={{ fontWeight: 'bold', color: 'orange' }}>
          Students with score below theta: {formatPercentage(cdfValue)}
        </span>
      </div>
      <div style={{ height: '300px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" domain={[-3.5, 3.5]} label={{ value: 'z-score', position: 'bottom', dy: 10 }} />
            <YAxis domain={[0, 0.5]} label={{ value: 'Probability Density', angle: -90, position: 'insideLeft', dx: -10 }} />
            <Tooltip formatter={value => value.toFixed(4)} labelFormatter={val => `z-score: ${val}`} />
            <Area type="monotone" dataKey="yArea" stroke="none" fill="#0000FF" fillOpacity={0.6} />
            <Line type="monotone" dataKey="y" stroke="#6f42c1" strokeWidth={2} dot={false} />
            <ReferenceLine x={theta} stroke="#000" strokeDasharray="4 4" label={{ value: '▼', position: 'top', fontSize: 14 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default NormalDistributionChart;

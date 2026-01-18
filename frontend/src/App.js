import React, { useState, useEffect, useRef } from 'react';
import { Package, AlertTriangle, Play, RotateCcw, Settings, Loader, Zap, Target, TrendingUp } from 'lucide-react';

const GRID_SIZE = 20;
const CELL_SIZE = 35;
const API_URL = 'http://localhost:8080/api';

const CELL_TYPES = {
  EMPTY: 0,
  SHELF: 1,
  OBSTACLE: 2,
  ROBOT: 3
};

const WarehouseRLFrontend = () => {
  const [warehouse, setWarehouse] = useState(null);
  const [robotPos, setRobotPos] = useState({ x: 0, y: 0 });
  const [selectedSKUs, setSelectedSKUs] = useState([]);
  const [path, setPath] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [avoidedShelves, setAvoidedShelves] = useState(new Set());
  const [showSettings, setShowSettings] = useState(false);
  const [visitedCells, setVisitedCells] = useState(new Set());
  const [error, setError] = useState(null);
  const [trainingProgress, setTrainingProgress] = useState(null);
  const animationRef = useRef(null);
  
  useEffect(() => {
    loadWarehouse();
  }, []);
  
  const loadWarehouse = async () => {
    try {
      const response = await fetch(`${API_URL}/warehouse/generate`);
      const data = await response.json();
      setWarehouse(data);
      setRobotPos({ x: 0, y: 0 });
      setError(null);
    } catch (err) {
      setError('Failed to connect to backend. Make sure Python server is running on port 5000.');
      console.error('Error loading warehouse:', err);
    }
  };
  
  const startOptimization = async () => {
    if (selectedSKUs.length === 0 || !warehouse) return;
    
    setIsTraining(true);
    setTrainingProgress('Initializing PPO training...');
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/warehouse/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selectedSKUs,
          avoidedShelves: Array.from(avoidedShelves)
        })
      });
      
      if (!response.ok) {
        throw new Error('Optimization failed');
      }
      
      const data = await response.json();
      setPath(data.path);
      setIsTraining(false);
      setTrainingProgress(null);
      setIsRunning(true);
      setCurrentStep(0);
      setVisitedCells(new Set());
      
      animatePath(data.path);
    } catch (err) {
      setError('Optimization failed. Check backend logs.');
      setIsTraining(false);
      setTrainingProgress(null);
      console.error('Error optimizing path:', err);
    }
  };
  
  const animatePath = (pathToAnimate) => {
    let step = 0;
    const visited = new Set();
    
    const animate = () => {
      if (step < pathToAnimate.length) {
        const [x, y] = pathToAnimate[step];
        setRobotPos({ x, y });
        visited.add(`${x},${y}`);
        setVisitedCells(new Set(visited));
        setCurrentStep(step);
        step++;
        animationRef.current = setTimeout(animate, 120);
      } else {
        setIsRunning(false);
      }
    };
    
    animate();
  };
  
  const reset = () => {
    if (animationRef.current) clearTimeout(animationRef.current);
    setIsRunning(false);
    setIsTraining(false);
    setPath([]);
    setCurrentStep(0);
    setRobotPos({ x: 0, y: 0 });
    setVisitedCells(new Set());
    setSelectedSKUs([]);
    setError(null);
    setTrainingProgress(null);
  };
  
  const toggleSKU = (sku) => {
    setSelectedSKUs(prev => 
      prev.includes(sku) ? prev.filter(s => s !== sku) : [...prev, sku]
    );
  };
  
  const toggleAvoidShelf = (shelf) => {
    const key = `${shelf.x},${shelf.y}`;
    setAvoidedShelves(prev => {
      const newSet = new Set(prev);
      if (newSet.has(key)) {
        newSet.delete(key);
      } else {
        newSet.add(key);
      }
      return newSet;
    });
  };
  
  if (!warehouse) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        {error ? (
          <div className="text-center bg-white rounded-2xl shadow-2xl p-8 max-w-md">
            <div className="text-red-600 mb-4 text-lg font-semibold">{error}</div>
            <div className="text-sm text-gray-600 mb-4 bg-gray-100 p-3 rounded-lg">
              <code className="text-xs">python backend/app.py</code>
            </div>
            <button 
              onClick={loadWarehouse}
              className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg"
            >
              Retry Connection
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-3 bg-white px-6 py-4 rounded-2xl shadow-2xl">
            <Loader className="animate-spin text-indigo-600" size={24} />
            <span className="text-lg font-semibold text-gray-800">Loading warehouse...</span>
          </div>
        )}
      </div>
    );
  }
  
  const getCellColor = (x, y) => {
    const cellType = warehouse.grid[y][x];
    const isRobot = robotPos.x === x && robotPos.y === y;
    const isPath = path.some(([px, py]) => px === x && py === y);
    const isVisited = visitedCells.has(`${x},${y}`);
    const isAvoided = avoidedShelves.has(`${x},${y}`);
    
    if (isRobot) return '#10b981';
    if (isAvoided) return '#ef4444';
    if (isPath && !isRobot) return '#6366f1';
    if (isVisited && !isRobot) return '#a5b4fc';
    if (cellType === CELL_TYPES.SHELF) return '#f59e0b';
    if (cellType === CELL_TYPES.OBSTACLE) return '#64748b';
    return '#f8fafc';
  };
  
  const efficiency = path.length > 0 ? Math.max(0, 100 - ((currentStep / path.length) * 100)) : 0;
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-2xl shadow-2xl mb-4">
            <Package size={32} />
            <h1 className="text-3xl font-bold">Warehouse RL Path Optimizer</h1>
          </div>
          <p className="text-purple-200 text-sm">
            Powered by PPO (Proximal Policy Optimization) + Stable-Baselines3
          </p>
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-3xl shadow-2xl overflow-hidden">
          {/* Control Panel */}
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 border-b-2 border-indigo-100">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-3">
                <div className="bg-white rounded-xl px-4 py-2 shadow-md">
                  <span className="text-sm text-gray-600">Selected Items:</span>
                  <span className="ml-2 text-lg font-bold text-indigo-600">{selectedSKUs.length}</span>
                </div>
                {path.length > 0 && !isTraining && (
                  <div className="bg-white rounded-xl px-4 py-2 shadow-md">
                    <span className="text-sm text-gray-600">Total Steps:</span>
                    <span className="ml-2 text-lg font-bold text-purple-600">{path.length}</span>
                  </div>
                )}
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className={`px-4 py-2 rounded-xl transition-all transform hover:scale-105 shadow-lg flex items-center gap-2 ${
                    showSettings 
                      ? 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white' 
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Settings size={20} />
                  <span className="hidden sm:inline">Settings</span>
                </button>
                <button
                  onClick={startOptimization}
                  disabled={isRunning || isTraining || selectedSKUs.length === 0}
                  className="px-6 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl hover:from-green-600 hover:to-emerald-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all transform hover:scale-105 shadow-lg flex items-center gap-2 font-semibold"
                >
                  {isTraining ? (
                    <>
                      <Loader className="animate-spin" size={20} />
                      Training...
                    </>
                  ) : (
                    <>
                      <Play size={20} />
                      Start
                    </>
                  )}
                </button>
                <button
                  onClick={reset}
                  className="px-6 py-2 bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl hover:from-red-600 hover:to-pink-700 transition-all transform hover:scale-105 shadow-lg flex items-center gap-2 font-semibold"
                >
                  <RotateCcw size={20} />
                  Reset
                </button>
              </div>
            </div>
            
            {/* Error Display */}
            {error && (
              <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-xl text-red-700 flex items-center gap-2">
                <AlertTriangle size={20} />
                {error}
              </div>
            )}
            
            {/* Training Progress */}
            {isTraining && (
              <div className="mt-4 p-4 bg-indigo-50 border-2 border-indigo-200 rounded-xl">
                <div className="flex items-center gap-3 mb-2">
                  <Loader className="animate-spin text-indigo-600" size={20} />
                  <span className="text-indigo-900 font-semibold">Training PPO Neural Network</span>
                </div>
                <div className="text-sm text-indigo-700">
                  Learning optimal pathfinding policy... This may take 10-20 seconds per item.
                </div>
                <div className="mt-3 bg-indigo-200 rounded-full h-2 overflow-hidden">
                  <div className="bg-gradient-to-r from-indigo-600 to-purple-600 h-full animate-pulse" style={{width: '100%'}}></div>
                </div>
              </div>
            )}
            
            {/* Settings Panel */}
            {showSettings && (
              <div className="mt-4 p-4 bg-yellow-50 border-2 border-yellow-200 rounded-xl">
                <h3 className="font-semibold mb-3 flex items-center gap-2 text-yellow-900">
                  <AlertTriangle className="text-yellow-600" size={20} />
                  Construction Zones (Click shelves to avoid)
                </h3>
                <div className="flex flex-wrap gap-2">
                  {warehouse.shelves.map(shelf => (
                    <button
                      key={shelf.id}
                      onClick={() => toggleAvoidShelf(shelf)}
                      className={`px-4 py-2 rounded-lg transition-all transform hover:scale-105 font-medium shadow-md ${
                        avoidedShelves.has(`${shelf.x},${shelf.y}`)
                          ? 'bg-gradient-to-r from-red-500 to-red-600 text-white'
                          : 'bg-white text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      {shelf.id}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {/* Main Grid */}
          <div className="p-8">
            <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
              {/* Warehouse Grid - Takes 3 columns */}
              <div className="xl:col-span-3 flex flex-col items-center">
                <div className="bg-gradient-to-br from-slate-100 to-slate-200 rounded-2xl p-6 shadow-inner">
                  <svg width={GRID_SIZE * CELL_SIZE} height={GRID_SIZE * CELL_SIZE} className="rounded-xl shadow-lg">
                    {/* Grid cells */}
                    {Array(GRID_SIZE).fill(null).map((_, y) =>
                      Array(GRID_SIZE).fill(null).map((_, x) => (
                        <rect
                          key={`${x}-${y}`}
                          x={x * CELL_SIZE}
                          y={y * CELL_SIZE}
                          width={CELL_SIZE}
                          height={CELL_SIZE}
                          fill={getCellColor(x, y)}
                          stroke="#cbd5e1"
                          strokeWidth="1.5"
                          rx="2"
                        />
                      ))
                    )}
                    
                    {/* Path line */}
                    {path.length > 1 && (
                      <path
                        d={path.map(([x, y], i) => 
                          `${i === 0 ? 'M' : 'L'} ${x * CELL_SIZE + CELL_SIZE/2} ${y * CELL_SIZE + CELL_SIZE/2}`
                        ).join(' ')}
                        stroke="#6366f1"
                        strokeWidth="3"
                        fill="none"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        opacity="0.6"
                      />
                    )}
                    
                    {/* Shelf labels */}
                    {warehouse.shelves.map(shelf => (
                      <text
                        key={shelf.id}
                        x={shelf.x * CELL_SIZE + CELL_SIZE / 2}
                        y={shelf.y * CELL_SIZE + CELL_SIZE / 2}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize="12"
                        fontWeight="bold"
                        fill="#78350f"
                      >
                        {shelf.id.split('_')[1]}
                      </text>
                    ))}
                    
                    {/* Robot with glow effect */}
                    <g>
                      <circle
                        cx={robotPos.x * CELL_SIZE + CELL_SIZE / 2}
                        cy={robotPos.y * CELL_SIZE + CELL_SIZE / 2}
                        r={CELL_SIZE / 2.5}
                        fill="#10b981"
                        opacity="0.3"
                      >
                        {isRunning && (
                          <animate
                            attributeName="r"
                            values={`${CELL_SIZE/2.5};${CELL_SIZE/1.8};${CELL_SIZE/2.5}`}
                            dur="1s"
                            repeatCount="indefinite"
                          />
                        )}
                      </circle>
                      <circle
                        cx={robotPos.x * CELL_SIZE + CELL_SIZE / 2}
                        cy={robotPos.y * CELL_SIZE + CELL_SIZE / 2}
                        r={CELL_SIZE / 3.5}
                        fill="#10b981"
                        stroke="#065f46"
                        strokeWidth="2.5"
                      />
                    </g>
                  </svg>
                </div>
                
                {/* Legend */}
                <div className="mt-6 flex items-center gap-6 text-sm flex-wrap justify-center">
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 bg-green-500 rounded-full shadow-md"></div>
                    <span className="font-medium text-gray-700">Robot</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 bg-amber-500 rounded shadow-md"></div>
                    <span className="font-medium text-gray-700">Shelf</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 bg-slate-500 rounded shadow-md"></div>
                    <span className="font-medium text-gray-700">Obstacle</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 bg-indigo-500 rounded shadow-md"></div>
                    <span className="font-medium text-gray-700">Path</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 bg-indigo-200 rounded shadow-md"></div>
                    <span className="font-medium text-gray-700">Visited</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 bg-red-500 rounded shadow-md"></div>
                    <span className="font-medium text-gray-700">Avoided</span>
                  </div>
                </div>
              </div>
              
              {/* Sidebar - Takes 1 column */}
              <div className="xl:col-span-1 space-y-6">
                {/* Pick List */}
                <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl p-5 shadow-lg border-2 border-indigo-100">
                  <h3 className="font-bold mb-4 text-lg flex items-center gap-2 text-indigo-900">
                    <Package size={20} />
                    Pick List
                  </h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                    {warehouse.shelves.map(shelf => (
                      <div key={shelf.id} className="bg-white rounded-xl p-3 shadow-md hover:shadow-lg transition-shadow">
                        <div className="font-semibold text-sm text-indigo-700 mb-2">{shelf.id}</div>
                        <div className="space-y-1.5">
                          {shelf.skus.map(sku => (
                            <label 
                              key={sku} 
                              className="flex items-center gap-2 cursor-pointer hover:bg-indigo-50 p-1.5 rounded-lg transition-colors"
                            >
                              <input
                                type="checkbox"
                                checked={selectedSKUs.includes(sku)}
                                onChange={() => toggleSKU(sku)}
                                className="w-4 h-4 text-indigo-600 rounded focus:ring-2 focus:ring-indigo-500"
                              />
                              <span className="text-sm font-medium text-gray-700">{sku}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Statistics */}
                {path.length > 0 && !isTraining && (
                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-5 shadow-lg border-2 border-green-100">
                    <h3 className="font-bold mb-4 text-lg flex items-center gap-2 text-green-900">
                      <TrendingUp size={20} />
                      Statistics
                    </h3>
                    <div className="space-y-3">
                      <div className="bg-white rounded-lg p-3 shadow-sm">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-gray-600">Progress</span>
                          <span className="font-bold text-green-600">{currentStep + 1} / {path.length}</span>
                        </div>
                        <div className="mt-2 w-full bg-gray-200 rounded-full h-2.5">
                          <div 
                            className="bg-gradient-to-r from-green-500 to-emerald-600 h-2.5 rounded-full transition-all"
                            style={{ width: `${((currentStep + 1) / path.length) * 100}%` }}
                          />
                        </div>
                      </div>
                      
                      <div className="bg-white rounded-lg p-3 shadow-sm">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-gray-600">Items Picked</span>
                          <span className="font-bold text-indigo-600">{selectedSKUs.length}</span>
                        </div>
                      </div>
                      
                      <div className="bg-white rounded-lg p-3 shadow-sm">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-gray-600">Efficiency</span>
                          <span className="font-bold text-purple-600">{Math.round(efficiency)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="mt-8 text-center">
          <div className="inline-block bg-white rounded-2xl shadow-xl px-6 py-4">
            <div className="flex items-center gap-3 text-sm">
              <Zap className="text-yellow-500" size={20} />
              <span className="text-gray-700">
                <strong className="text-indigo-600">Backend:</strong> Python Flask + Stable-Baselines3 PPO
              </span>
              <span className="text-gray-300">|</span>
              <span className="text-gray-700">
                <strong className="text-purple-600">Frontend:</strong> React.js
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WarehouseRLFrontend;
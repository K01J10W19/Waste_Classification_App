const STORAGE_KEY = "waste_scan_history";

export const saveScan = (scan) => {
  const history = getHistory();
  history.unshift(scan);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(0, 100)));
};

export const getHistory = () => {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? [];
  } catch {
    return [];
  }
};

export const clearHistory = () => localStorage.removeItem(STORAGE_KEY);

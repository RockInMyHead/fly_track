const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('flytrack', {
  isDesktop: true,
  version: ipcRenderer.sendSync('app:version'),
  pickVideos: () => ipcRenderer.invoke('video-file:pick'),
  openDataFolder: () => ipcRenderer.invoke('app:open-data-folder'),
  openLogs: () => ipcRenderer.invoke('app:open-logs'),
});

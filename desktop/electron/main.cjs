const { app, BrowserWindow, Menu, dialog, shell, ipcMain } = require('electron');
const fs = require('fs');
const path = require('path');
const { createBackend } = require('./backend.cjs');

const TITLE = 'Fly Track';
let mainWindow = null;
let splash = null;
let backend = null;
let baseUrl = null;
let quitting = false;

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

if (process.platform === 'win32') app.setAppUserModelId('com.flytrack.app');

function createSplash() {
  splash = new BrowserWindow({
    width: 420,
    height: 300,
    frame: false,
    resizable: false,
    transparent: false,
    backgroundColor: '#07111f',
    show: true,
    icon: path.join(__dirname, 'icon.png'),
  });
  splash.loadFile(path.join(__dirname, 'splash.html'));
}

function isInternal(url) {
  return Boolean(baseUrl) && url.startsWith(baseUrl);
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    show: false,
    title: TITLE,
    backgroundColor: '#07111f',
    icon: path.join(__dirname, 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (!isInternal(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!isInternal(url)) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });
  mainWindow.once('ready-to-show', () => {
    if (splash && !splash.isDestroyed()) splash.close();
    splash = null;
    mainWindow.show();
    maybeSelfCheck();
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
  const clip = process.env.FLY_OPEN_CLIP;
  mainWindow.loadURL(`${baseUrl}/ui/${clip ? `?clip=${encodeURIComponent(clip)}` : ''}`);
}

// FLY_SHOT=file.png: save a picture of the window after it settles (CI and review).
function maybeSelfCheck() {
  const shot = process.env.FLY_SHOT;
  if (!shot) return;
  const delay = Number(process.env.FLY_SHOT_DELAY_MS || 6000);
  setTimeout(async () => {
    try {
      const image = await mainWindow.webContents.capturePage();
      fs.writeFileSync(shot, image.toPNG());
      console.log(`SHOT ${shot}`);
    } catch (e) {
      console.error(`SHOT FAILED ${e.message}`);
    }
    if (process.env.FLY_SHOT_QUIT === '1') app.quit();
  }, delay);
}

function createMenu() {
  const L = backend.layout;
  Menu.setApplicationMenu(
    Menu.buildFromTemplate([
      {
        label: TITLE,
        submenu: [
          {
            label: 'О программе',
            click: () =>
              dialog.showMessageBox(mainWindow, {
                type: 'info',
                title: TITLE,
                message: `Fly Track ${app.getVersion()}`,
                detail:
                  'Видео с камеры → анализ → трекер на графе цеха (V1–V5).\n' +
                  `Данные: ${L.workspace}`,
              }),
          },
          { label: 'Открыть папку с данными', click: () => shell.openPath(L.workspace) },
          { label: 'Открыть журналы', click: () => shell.openPath(backend.logsDir) },
          { type: 'separator' },
          { role: 'quit', label: 'Выход' },
        ],
      },
      {
        label: 'Вид',
        submenu: [
          { role: 'reload', label: 'Обновить' },
          { type: 'separator' },
          { role: 'resetZoom', label: 'Сбросить масштаб' },
          { role: 'zoomIn', label: 'Увеличить' },
          { role: 'zoomOut', label: 'Уменьшить' },
          { role: 'togglefullscreen', label: 'Полный экран' },
          ...(app.isPackaged ? [] : [{ role: 'toggleDevTools', label: 'Инструменты разработчика' }]),
        ],
      },
    ]),
  );
}

ipcMain.on('app:version', (event) => {
  event.returnValue = app.getVersion();
});
ipcMain.handle('video-file:pick', async (event) => {
  const owner = BrowserWindow.fromWebContents(event.sender) || mainWindow;
  const { canceled, filePaths } = await dialog.showOpenDialog(owner, {
    title: 'Выберите видео с камеры (AVI)',
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'Видео камеры', extensions: ['avi', 'AVI'] }],
  });
  return canceled ? [] : filePaths;
});
ipcMain.handle('app:open-data-folder', () => shell.openPath(backend.layout.workspace));
ipcMain.handle('app:open-logs', () => shell.openPath(backend.logsDir));

async function boot() {
  createSplash();
  backend = createBackend({
    onExit: async (code) => {
      if (quitting) return;
      const { response } = await dialog.showMessageBox(mainWindow, {
        type: 'error',
        title: TITLE,
        message: 'Локальный расчёт остановился.',
        detail: `Код ${code}. Журнал: ${backend.logFile}`,
        buttons: ['Перезапустить', 'Закрыть'],
      });
      if (response === 0) {
        app.relaunch();
      }
      app.quit();
    },
  });
  createMenu();
  try {
    baseUrl = await backend.start();
    console.log(`BACKEND ${baseUrl}`);
  } catch (e) {
    if (splash && !splash.isDestroyed()) splash.close();
    await dialog.showMessageBox({
      type: 'error',
      title: TITLE,
      message: 'Не удалось запустить Fly Track.',
      detail: `${e.message}\n\nЖурнал: ${backend.logFile}`,
    });
    quitting = true;
    backend.stop();
    app.exit(1);
    return;
  }
  createWindow();
}

app.whenReady().then(boot);

app.on('before-quit', () => {
  quitting = true;
  if (backend) backend.stop();
});

app.on('window-all-closed', () => {
  app.quit();
});

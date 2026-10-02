; Fly Track installer (Inno Setup 6).
;   iscc /DStage=dist\win\FlyTrack /DAppVersion=1.0.0 desktop\windows\installer.iss
; Stage holds python\, ffmpeg\ and app\ (see .github/workflows/windows-app.yml).
; Per-user install: the app folder must stay writable (camera videos, brain records, routes).

#ifndef Stage
  #define Stage "..\..\dist\win\FlyTrack"
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B0F3C1E-2A57-4C8E-9F1D-7A4E2B9C5D31}
AppName=Fly Track
AppVersion={#AppVersion}
AppPublisher=Fly Track
DefaultDirName={localappdata}\Programs\FlyTrack
DefaultGroupName=Fly Track
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=FlyTrack-Setup-{#AppVersion}
SetupIconFile=..\assets\fly_track.ico
UninstallDisplayIcon={app}\app\desktop\assets\fly_track.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app\output\numba_cache"

[Files]
Source: "{#Stage}\python\*"; DestDir: "{app}\python"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Stage}\ffmpeg\*"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Stage}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Fly Track"; Filename: "{app}\python\pythonw.exe"; \
  Parameters: "-X utf8 ""{app}\app\desktop\fly_track_app.py"""; WorkingDir: "{app}\app"; \
  IconFilename: "{app}\app\desktop\assets\fly_track.ico"; AppUserModelID: "FlyTrack.App"
Name: "{autodesktop}\Fly Track"; Filename: "{app}\python\pythonw.exe"; \
  Parameters: "-X utf8 ""{app}\app\desktop\fly_track_app.py"""; WorkingDir: "{app}\app"; \
  IconFilename: "{app}\app\desktop\assets\fly_track.ico"; AppUserModelID: "FlyTrack.App"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-X utf8 ""{app}\app\desktop\fly_track_app.py"""; \
  WorkingDir: "{app}\app"; Description: "{cm:LaunchProgram,Fly Track}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app\output\numba_cache"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if DirExists(ExpandConstant('{app}\app\data\app')) then
      if MsgBox('Удалить и загруженные с камеры видео, записи мозга и маршруты?' + #13#10 +
                ExpandConstant('{app}\app'), mbConfirmation, MB_YESNO or MB_DEFAULTBUTTON2) = IDYES then
        DelTree(ExpandConstant('{app}'), True, True, True);
end;

/** Общая синхронизация: время видео → панель мухи → точка на плане. */
(function () {
  function sampleTraj(rows, t) {
    if (!rows || !rows.length) return null;
    let i = rows.findIndex(r => +r.time >= t);
    if (i < 0) i = rows.length - 1;
    if (i > 0 && +rows[i].time - t > t - +rows[i - 1].time) i--;
    return rows[i];
  }

  function bindVideo(video, onTime) {
    if (!video || !onTime) return;
    const tick = () => onTime(video.currentTime || 0, 'video');
    video.addEventListener('timeupdate', tick);
    video.addEventListener('seeked', tick);
    video.addEventListener('play', tick);
  }

  window.PlaybackSync = { sampleTraj, bindVideo };
})();

(async () => {
  try {
    await window.magpie.ready;
    if (typeof window.magpie.service !== 'string' || !/^https?:\/\//.test(window.magpie.service)) {
      throw new Error('当前 Magpie 不支持远程插件服务。');
    }
    location.replace(window.magpie.service);
  } catch (error) {
    document.getElementById('message').textContent = error.message || '无法连接插件服务。';
  }
})();

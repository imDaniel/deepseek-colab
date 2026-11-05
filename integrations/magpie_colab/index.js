// The service runs independently under systemd; this host hook declares the
// plugin's service to Magpie without requiring it to manage Colab itself.
export default async function ColabDeepSeek() {
  return {};
}

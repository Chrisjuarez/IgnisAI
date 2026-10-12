describe('prediction API client', () => {
  let client;
  let predictFireSpreadMultistep;

  beforeEach(() => {
    jest.resetModules();
    client = { get: jest.fn(), interceptors: { response: { use: jest.fn() } } };
    jest.doMock('axios', () => ({
      __esModule: true,
      default: {
        create: jest.fn(() => client),
      },
    }));
    ({ predictFireSpreadMultistep } = require('../api'));
  });

  afterEach(() => {
    jest.dontMock('axios');
  });

  test('does not send a threshold override by default', async () => {
    client.get.mockResolvedValueOnce({ data: { ok: true } });

    await predictFireSpreadMultistep({
      lat: 34.05,
      lon: -118.55,
      steps: 6,
      stepHours: 24,
      date: '2025-01-07T18:30:00Z',
    });

    expect(client.get).toHaveBeenCalledWith(
      '/predict-fire-spread/multistep',
      expect.objectContaining({
        params: expect.not.objectContaining({ thr: expect.anything() }),
      }),
    );
  });

  test('sends threshold only when explicitly supplied', async () => {
    client.get.mockResolvedValueOnce({ data: { ok: true } });

    await predictFireSpreadMultistep({
      lat: 34.05,
      lon: -118.55,
      thr: 0.42,
    });

    expect(client.get).toHaveBeenCalledWith(
      '/predict-fire-spread/multistep',
      expect.objectContaining({
        params: expect.objectContaining({ thr: 0.42 }),
      }),
    );
  });

  test('sends ignition only when explicitly supplied', async () => {
    client.get.mockResolvedValueOnce({ data: { ok: true } });

    await predictFireSpreadMultistep({
      lat: 34.05,
      lon: -118.55,
      ignition: false,
    });

    expect(client.get).toHaveBeenCalledWith(
      '/predict-fire-spread/multistep',
      expect.objectContaining({
        params: expect.objectContaining({ ignition: false }),
      }),
    );
  });
});

describe('failed requests', () => {
  let client;
  let readableMessage;
  let attachReadableMessage;

  beforeEach(() => {
    jest.resetModules();
    client = { get: jest.fn(), interceptors: { response: { use: jest.fn() } } };
    jest.doMock('axios', () => ({ __esModule: true, default: { create: jest.fn(() => client) } }));
    ({ readableMessage, attachReadableMessage } = require('../api'));
  });

  afterEach(() => {
    jest.dontMock('axios');
  });

  const AXIOS_MESSAGE = 'Request failed (axios default text)';
  const failed = (props) => Object.assign(new Error(AXIOS_MESSAGE), props);

  test('carry the message the API sent', () => {
    const error = failed({ response: { status: 422, data: { message: 'Terrain and fuel data for this area could not be loaded.' } } });
    expect(readableMessage(error)).toBe('Terrain and fuel data for this area could not be loaded.');
  });

  test('say the server is having trouble when it failed without a message', () => {
    // Render's own error page while the backend restarts is HTML, not our JSON.
    const error = failed({ response: { status: 502, data: '<html>Bad Gateway</html>' } });
    expect(readableMessage(error)).toBe('Ignis is having trouble right now. Try again shortly.');
  });

  test('say a timeout took too long', () => {
    expect(readableMessage(failed({ code: 'ECONNABORTED' }))).toMatch(/took too long/);
  });

  test('say when Ignis could not be reached', () => {
    expect(readableMessage(failed({ code: 'ERR_NETWORK' }))).toMatch(/could not be reached/);
  });

  test('have nothing better to say about a 4xx without a message', () => {
    expect(readableMessage(failed({ response: { status: 404, data: '' } }))).toBeUndefined();
  });

  test('keep their own message when there is nothing better', () => {
    const error = failed({ response: { status: 404, data: '' } });
    expect(attachReadableMessage(error).message).toBe(AXIOS_MESSAGE);
  });

  test('get the readable message on every request through the client', async () => {
    const [, onRejected] = client.interceptors.response.use.mock.calls[0];
    const error = failed({ response: { data: { message: 'The forecast took too long. Try again in a minute.' } } });

    await expect(onRejected(error)).rejects.toThrow('The forecast took too long. Try again in a minute.');
  });
});

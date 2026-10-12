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
  let readableError;

  beforeEach(() => {
    jest.resetModules();
    client = { get: jest.fn(), interceptors: { response: { use: jest.fn() } } };
    jest.doMock('axios', () => ({ __esModule: true, default: { create: jest.fn(() => client) } }));
    ({ readableError } = require('../api'));
  });

  afterEach(() => {
    jest.dontMock('axios');
  });

  const failed = (props) => Object.assign(new Error('Request failed with status code 502'), props);

  test('carry the message the API sent', () => {
    const error = failed({ response: { status: 422, data: { message: 'Terrain and fuel data for this area could not be loaded.' } } });
    expect(readableError(error).message).toBe('Terrain and fuel data for this area could not be loaded.');
  });

  test('say a timeout took too long', () => {
    expect(readableError(failed({ code: 'ECONNABORTED' })).message).toMatch(/took too long/);
  });

  test('say when Ignis could not be reached', () => {
    expect(readableError(failed({ code: 'ERR_NETWORK' })).message).toMatch(/could not be reached/);
  });

  test('keep their own message when there is nothing better', () => {
    expect(readableError(failed({ response: { status: 500, data: 'oops' } })).message).toBe('Request failed with status code 502');
  });

  test('pass through the client for every request', async () => {
    const [, onRejected] = client.interceptors.response.use.mock.calls[0];
    const error = failed({ response: { data: { message: 'The forecast took too long. Try again in a minute.' } } });

    await expect(onRejected(error)).rejects.toThrow('The forecast took too long. Try again in a minute.');
  });
});

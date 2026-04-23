describe('prediction API client', () => {
  let client;
  let predictFireSpreadMultistep;

  beforeEach(() => {
    jest.resetModules();
    client = { get: jest.fn() };
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
});

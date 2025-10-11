const request = require('supertest');
const app = require('../app');
const User = require('../models/User');

// Mock the User model
jest.mock('../models/User');

describe('Authentication Integration Tests', () => {
  let server;
  
  beforeAll(async () => {
    server = app.listen(0);
  });

  afterAll(async () => {
    if (server) {
      await new Promise((res) => server.close(res));
    }
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('POST /api/auth/register', () => {
    it('TC-001: should register new user successfully', async () => {
      // Mock User methods
      User.findOne.mockResolvedValue(null); // No existing user
      User.prototype.save = jest.fn().mockResolvedValue({
        _id: 'mock-user-id',
        fullName: 'John Doe',
        email: 'john.doe@example.com',
        phone: '(555) 123-4567'
      });
      User.mockImplementation(() => ({
        _id: 'mock-user-id',
        fullName: 'John Doe',
        email: 'john.doe@example.com',
        phone: '(555) 123-4567',
        save: User.prototype.save
      }));

      const userData = {
        fullName: 'John Doe',
        email: 'john.doe@example.com',
        phone: '(555) 123-4567',
        password: 'password123'
      };

      const response = await request(server)
        .post('/api/auth/register')
        .send(userData)
        .expect(201);

      expect(response.body.message).toBe('User registered successfully');
      expect(response.body.user.email).toBe(userData.email);
      expect(response.body.token).toBeDefined();
      expect(User.findOne).toHaveBeenCalledWith({ email: { $eq: userData.email } });
    });

    it('TC-002: should reject duplicate email registration', async () => {
      // Mock existing user
      User.findOne.mockResolvedValue({
        email: 'john.doe@example.com',
        fullName: 'Existing User'
      });

      const response = await request(app)
        .post('/api/auth/register')
        .send({
          fullName: 'Jane Doe',
          email: 'john.doe@example.com',
          phone: '(555) 987-6543',
          password: 'password456'
        })
        .expect(400);

      expect(response.body.message).toBe('Email already registered');
    });

    it('TC-003: should reject invalid input data', async () => {
      const response = await request(app)
        .post('/api/auth/register')
        .send({
          fullName: '',
          email: 'invalid-email',
          password: '123'
        })
        .expect(400);

      expect(response.body.message).toBe('Invalid input data');
    });
  });

  describe('POST /api/auth/login', () => {
    it('TC-004: should login with valid credentials', async () => {
      // Mock user with comparePassword method
      const mockUser = {
        _id: 'mock-user-id',
        fullName: 'John Doe',
        email: 'john.doe@example.com',
        phone: '(555) 123-4567',
        comparePassword: jest.fn().mockResolvedValue(true),
        save: jest.fn().mockResolvedValue()
      };

      User.findOne.mockReturnValue({
        select: jest.fn().mockResolvedValue(mockUser)
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'john.doe@example.com',
          password: 'password123'
        })
        .expect(200);

      expect(response.body.message).toBe('Login successful');
      expect(response.body.user.email).toBe('john.doe@example.com');
      expect(response.body.token).toBeDefined();
    });

    it('TC-005: should reject invalid credentials', async () => {
      User.findOne.mockReturnValue({
        select: jest.fn().mockResolvedValue(null)
      });

      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'john.doe@example.com',
          password: 'wrongpassword'
        })
        .expect(401);

      expect(response.body.message).toBe('Invalid email or password');
    });
  });

  describe('POST /api/auth/forgot-password', () => {
    it('TC-006: should generate reset token for valid email', async () => {
      const mockUser = {
        createPasswordResetToken: jest.fn().mockReturnValue('mock-reset-token'),
        save: jest.fn().mockResolvedValue()
      };

      User.findOne.mockResolvedValue(mockUser);

      const response = await request(app)
        .post('/api/auth/forgot-password')
        .send({
          email: 'john.doe@example.com'
        })
        .expect(200);

      expect(response.body.message).toBe('Password reset link sent to your email');
    });

    it('TC-007: should reject unknown email', async () => {
      User.findOne.mockResolvedValue(null);

      const response = await request(app)
        .post('/api/auth/forgot-password')
        .send({
          email: 'unknown@example.com'
        })
        .expect(404);

      expect(response.body.message).toBe('No account found with this email');
    });
  });
});
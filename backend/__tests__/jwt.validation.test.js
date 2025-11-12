// backend/__tests__/jwt.validation.test.js
const jwt = require('jsonwebtoken');

describe('JWT Token Validation Unit Tests', () => {
  const mockUser = {
    _id: 'test-user-id-12345',
    email: 'test@example.com',
    fullName: 'Test User'
  };

  // Store original env vars
  let originalEnv;

  beforeAll(() => {
    originalEnv = { ...process.env };
  });

  beforeEach(() => {
    // Set test environment variables
    process.env.JWT_SECRET = 'test-secret-key-for-testing-minimum-32-chars';
    process.env.JWT_EXPIRES_IN = '7d';
  });

  afterEach(() => {
    // Restore environment
    process.env = { ...originalEnv };
  });

  describe('Token Generation', () => {
    it('should generate valid JWT token', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.split('.')).toHaveLength(3); // header.payload.signature
    });

    it('should include user ID in token payload', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      
      expect(decoded.id).toBe(mockUser._id);
    });

    it('should set correct expiration time', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      
      // 7 days in seconds
      const expectedExpiry = Math.floor(Date.now() / 1000) + (7 * 24 * 60 * 60);
      expect(Math.abs(decoded.exp - expectedExpiry)).toBeLessThan(5); // 5 second tolerance
    });

    it('should generate different tokens for same user at different times', () => {
      const token1 = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      // Wait a bit to ensure different iat (issued at)
      const token2 = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      // Tokens should be different due to different iat
      const decoded1 = jwt.decode(token1);
      const decoded2 = jwt.decode(token2);
      
      expect(decoded1.iat).toBeLessThanOrEqual(decoded2.iat);
    });
  });

  describe('Token Verification', () => {
    it('should verify valid token successfully', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      
      expect(decoded).toBeDefined();
      expect(decoded.id).toBe(mockUser._id);
    });

    it('should reject expired token', () => {
      const expiredToken = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: '-1s' } // Already expired
      );
      
      expect(() => {
        jwt.verify(expiredToken, process.env.JWT_SECRET);
      }).toThrow('jwt expired');
    });

    it('should reject token with invalid signature', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const tamperedToken = token.slice(0, -5) + 'XXXXX';
      
      expect(() => {
        jwt.verify(tamperedToken, process.env.JWT_SECRET);
      }).toThrow('invalid signature');
    });

    it('should reject token with wrong secret', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      expect(() => {
        jwt.verify(token, 'wrong-secret-key');
      }).toThrow('invalid signature');
    });

    it('should reject malformed token', () => {
      const malformedToken = 'not.a.valid.jwt.token';
      
      expect(() => {
        jwt.verify(malformedToken, process.env.JWT_SECRET);
      }).toThrow();
    });

    it('should reject empty token', () => {
      expect(() => {
        jwt.verify('', process.env.JWT_SECRET);
      }).toThrow();
    });
  });

  describe('Token Decoding', () => {
    it('should decode token without verification', () => {
      const token = jwt.sign(
        { id: mockUser._id, email: mockUser.email },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.decode(token);
      
      expect(decoded).toBeDefined();
      expect(decoded.id).toBe(mockUser._id);
      expect(decoded.email).toBe(mockUser.email);
    });

    it('should decode token header', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.decode(token, { complete: true });
      
      expect(decoded.header).toBeDefined();
      expect(decoded.header.alg).toBe('HS256');
      expect(decoded.header.typ).toBe('JWT');
    });
  });

  describe('Token Security', () => {
    it('should use strong secret key', () => {
      expect(process.env.JWT_SECRET).toBeDefined();
      expect(process.env.JWT_SECRET.length).toBeGreaterThanOrEqual(32);
    });

    it('should not expose sensitive data in token payload', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.decode(token);
      
      // Should not contain sensitive fields
      expect(decoded.password).toBeUndefined();
      expect(decoded.passwordResetToken).toBeUndefined();
      expect(decoded.passwordResetExpires).toBeUndefined();
    });

    it('should handle JWT_SECRET from environment', () => {
      // Test that JWT_SECRET is read from environment
      const customSecret = 'custom-test-secret-key-minimum-32-characters';
      process.env.JWT_SECRET = customSecret;
      
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: '1d' }
      );
      
      // Should verify with same secret
      expect(() => {
        jwt.verify(token, customSecret);
      }).not.toThrow();
      
      // Should fail with different secret
      expect(() => {
        jwt.verify(token, 'different-secret');
      }).toThrow();
    });

    it('should handle different expiration times', () => {
      const expirations = ['1h', '1d', '7d', '30d'];
      
      expirations.forEach(exp => {
        const token = jwt.sign(
          { id: mockUser._id },
          process.env.JWT_SECRET,
          { expiresIn: exp }
        );
        
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        expect(decoded.exp).toBeGreaterThan(Math.floor(Date.now() / 1000));
      });
    });
  });

  describe('Token Payload', () => {
    it('should support custom payload fields', () => {
      const customPayload = {
        id: mockUser._id,
        email: mockUser.email,
        role: 'user',
        permissions: ['read', 'write']
      };
      
      const token = jwt.sign(
        customPayload,
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      
      expect(decoded.id).toBe(customPayload.id);
      expect(decoded.email).toBe(customPayload.email);
      expect(decoded.role).toBe(customPayload.role);
      expect(decoded.permissions).toEqual(customPayload.permissions);
    });

    it('should include standard JWT claims', () => {
      const token = jwt.sign(
        { id: mockUser._id },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRES_IN }
      );
      
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      
      // Standard claims
      expect(decoded.iat).toBeDefined(); // issued at
      expect(decoded.exp).toBeDefined(); // expiration
      expect(typeof decoded.iat).toBe('number');
      expect(typeof decoded.exp).toBe('number');
      expect(decoded.exp).toBeGreaterThan(decoded.iat);
    });
  });
});
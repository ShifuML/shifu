/*
 * Copyright [2013-2019] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;

import java.util.List;

/**
 * Helper class for check the runtime environment should meet certain requirements, or the
 * {@link ShifuErrorCode#ASSERT_ERROR} type exception will be throw.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class AssertUtils {

    /**
     * Protect constructor since it is a static only class
     */
    protected AssertUtils() {
    }

    /**
     * Asserts that two ints are equal.
     *
     * @param expected, the expected value
     * @param actual,   the actual value
     */
    public static void assertEquals(int expected, int actual) {
        assertEquals(null, expected, actual);
    }

    /**
     * Asserts that two ints are equal. If they are not an AssertionFailedError is thrown with the given message.
     *
     * @param message,  failed message
     * @param expected, the expected value
     * @param actual,   the actual value
     */
    public static void assertEquals(String message, int expected, int actual) {
        assertEquals(message, new Integer(expected), new Integer(actual));
    }

    /**
     * Assert if two lists is not null and their size are equal
     *
     * @param a, the first list
     * @param b, the second list
     */
    @SuppressWarnings("rawtypes")
    public static void assertListNotNullAndSizeEqual(List a, List b) {
        assertNotNull(a);
        assertNotNull(b);
        assertEquals(a.size(), b.size());
    }

    /**
     * Assert if two float array is not null and their size are equal
     *
     * @param a, the first array
     * @param b, the second array
     */
    public static void assertFloatArrayNotNullAndLengthEqual(float[] a, float[] b) {
        assertNotNull(a);
        assertNotNull(b);
        assertEquals(a.length, b.length);
    }

    /**
     * Asserts that two objects are equal. If they are not
     * an AssertionFailedError is thrown with the given message.
     *
     * @param message,  failed messge
     * @param expected, expected value
     * @param actual,   the actual value
     */
    public static void assertEquals(String message, Object expected, Object actual) {
        if(expected == null && actual == null) {
            return;
        }
        if(expected != null && expected.equals(actual)) {
            return;
        }
        failNotEquals(message, expected, actual);
    }

    /**
     * Fails a test with the given message.
     *
     * @param message, the given message
     */
    public static void fail(String message) {
        if(message == null) {
            throw new ShifuException(ShifuErrorCode.ASSERT_ERROR);
        }
        throw new ShifuException(ShifuErrorCode.ASSERT_ERROR, message);
    }

    /**
     * Fails a test with no message.
     */
    @SuppressWarnings("all")
    public static void fail() {
        fail(null);
    }

    /**
     * Asserts that a condition is true. If it isn't it throws
     * an AssertionFailedError with the given message.
     *
     * @param message,   the given message
     * @param condition, check condition
     */
    @SuppressWarnings("all")
    public static void assertTrue(String message, boolean condition) {
        if(!condition) {
            fail(message);
        }
    }

    /**
     * Asserts that a condition is true. If it isn't it throws
     * an AssertionFailedError.
     *
     * @param condition, check condition
     */
    public static void assertTrue(boolean condition) {
        assertTrue(null, condition);
    }

    /**
     * Asserts that two objects are equal. If they are not
     * an AssertionFailedError is thrown.
     *
     * @param expected, expected object
     * @param actual,   check object
     */
    public static void assertEquals(Object expected, Object actual) {
        assertEquals(null, expected, actual);
    }

    /**
     * Asserts that two Strings are equal.
     *
     * @param message,  the given message
     * @param expected, the given message
     * @param actual,   the given message
     */
    public static void assertEquals(String message, String expected, String actual) {
        if(expected == null && actual == null) {
            return;
        }
        if(expected != null && expected.equals(actual)) {
            return;
        }
        String cleanMessage = message == null ? "" : message;
        throw new ShifuException(ShifuErrorCode.ASSERT_ERROR, cleanMessage);
    }

    /**
     * Asserts that two Strings are equal.
     *
     * @param expected, the expected value
     * @param actual,   the actual value
     */
    public static void assertEquals(String expected, String actual) {
        assertEquals(null, expected, actual);
    }

    /**
     * Asserts that an object isn't null.
     *
     * @param object, the object to evaluate
     */
    public static void assertNotNull(Object object) {
        assertNotNull(null, object);
    }

    /**
     * Asserts that an object isn't null. If it is an AssertionFailedError is thrown with the given message.
     *
     * @param message, the fail message
     * @param object,  the object to check
     */
    public static void assertNotNull(String message, Object object) {
        assertTrue(message, object != null);
    }

    /**
     * Asserts that an object is null. If it isn't an {@link AssertionError} is
     * thrown.
     * Message contains: Expected: null but was: object
     *
     * @param object Object to check
     */
    public static void assertNull(Object object) {
        String message = "Expected: <null> but was: " + String.valueOf(object);
        assertNull(message, object);
    }

    /**
     * Asserts that an object is null.  If it is not
     *
     * @param message, the message when check fail an AssertionFailedError is thrown with the given message.
     * @param object,  Object to check or null
     */
    public static void assertNull(String message, Object object) {
        assertTrue(message, object == null);
    }

    private static String format(String message, Object expected, Object actual) {
        String formatted = "";
        if(message != null && message.length() > 0) {
            formatted = message + " ";
        }
        return formatted + "expected:<" + expected + "> but was:<" + actual + ">";
    }

    @SuppressWarnings("all")
    private static void failNotEquals(String message, Object expected, Object actual) {
        fail(format(message, expected, actual));
    }

}

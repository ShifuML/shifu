/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.util;

import java.io.IOException;
import java.io.UnsupportedEncodingException;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.lang.StringUtils;

/**
 * Base64Utils class
 */
public class Base64Utils {
    private Base64Utils() {
        // singleton
    }

    public static String base64Encode(String text) {
        if(text == null) {
            return null;
        }

        Base64 encoder = new Base64(-1);
        try {
            return new String(encoder.encode(text.getBytes(Constants.DEFAULT_CHARSET)), Constants.DEFAULT_CHARSET);
        } catch (UnsupportedEncodingException e) {
            return null;
        }
    }

    public static String base64Decode(String text) throws IOException {
        if(text == null) {
            return null;
        }

        Base64 decoder = new Base64(-1);
        return new String(decoder.decode(text.getBytes(Constants.DEFAULT_CHARSET)), Constants.DEFAULT_CHARSET);
    }

    public static String base64DecodeIfEncodedInput(String text) throws IOException {
        return isBase64Encode(text) ? Base64Utils.base64Decode(text) : text;
    }

    public static boolean isBase64Encode(String text) throws IOException {
        String originInput = Base64Utils.base64Decode(text);
        return StringUtils.equals(text, Base64Utils.base64Encode(originInput));
    }

    public static String base64EncodeFromBytes(byte[] bytes) {
        if(bytes == null) {
            return null;
        }

        Base64 encoder = new Base64(-1);
        try {
            return new String(encoder.encode(bytes), Constants.DEFAULT_CHARSET);
        } catch (UnsupportedEncodingException e) {
            return null;
        }
    }

    public static byte[] base64DecodeToBytes(String bytes) throws UnsupportedEncodingException {
        if(bytes == null) {
            return null;
        }

        Base64 decoder = new Base64(-1);
        try {
            return decoder.decode(bytes.getBytes(Constants.DEFAULT_CHARSET));
        } catch (UnsupportedEncodingException e) {
            return null;
        }
    }
}

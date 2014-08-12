/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.core.util;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;

/**
 * <p/>
 * {@link JSONUtils} is a unified entry for all json format serialization and de-serialization.
 * <p/>
 * <p/>
 * ObjectMapper instance is stored into ThreadLocal object to make sure thread safety.
 */
public class JSONUtils {

    private static final ThreadLocal<ObjectMapper> jsonMapper = new ThreadLocal<ObjectMapper>() {
        @Override
        protected ObjectMapper initialValue() {
            return new ObjectMapper();
        }
    };

    private static ObjectMapper getObjectMapperInstance() {
        return jsonMapper.get();
    }

    /*
     * @see ObjectMapper#readValue(Reader, Class);
     */
    public static <T> T readValue(Reader src, Class<T> valueType) throws
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /*
     * @see ObjectMapper#readValue(File, Class);
     */
    public static <T> T readValue(File src, Class<T> valueType) throws
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /*
     * @see ObjectMapper#readValue(InputStream, Class);
     */
    public static <T> T readValue(InputStream src, Class<T> valueType) throws
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /*
     * @see ObjectMapper#readValue(src, Class);
     */
    public static <T> T readValue(String src, Class<T> valueType) throws
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /*
     * @see ObjectWriter#writeValue(Writer, Object);
     */
    public static void writeValue(Writer w, Object value) throws
            IOException {
        getObjectMapperInstance().writerWithDefaultPrettyPrinter().writeValue(w, value);
    }

    /*
     * @see ObjectWriter#writeValue(File, Object);
     */
    public static void writeValue(File src, Object value) throws
            IOException {
        getObjectMapperInstance().writerWithDefaultPrettyPrinter().writeValue(src, value);
    }

}

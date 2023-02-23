/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.util;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.io.Writer;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * {@link JSONUtils} is a unified entry for all json format serialization and de-serialization.
 * 
 * <p>
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
    public static <T> T readValue(Reader src, Class<T> valueType) throws JsonParseException, JsonMappingException,
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /*
     * @see ObjectMapper#readValue(File, Class);
     */
    public static <T> T readValue(File src, Class<T> valueType) throws JsonParseException, JsonMappingException,
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /*
     * @see ObjectMapper#readValue(InputStream, Class);
     */
    public static <T> T readValue(InputStream src, Class<T> valueType) throws JsonParseException, JsonMappingException,
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /*
     * @see ObjectMapper#readValue(src, Class);
     */
    public static <T> T readValue(String src, Class<T> valueType) throws JsonParseException, JsonMappingException,
            IOException {
        return getObjectMapperInstance().readValue(src, valueType);
    }

    /**
     * 
     * @param <T>
     * @param src
     * @param typeRef
     * @return
     * @throws JsonParseException
     * @throws JsonMappingException
     * @throws IOException
     */
    public static <T> T readValue(File src, TypeReference<T> typeRef) throws JsonParseException, JsonMappingException,
            IOException {
        return getObjectMapperInstance().readValue(src, typeRef);
    }

    /*
     * @see ObjectWriter#writeValue(Writer, Object);
     */
    public static void writeValue(Writer w, Object value) throws JsonGenerationException, JsonMappingException,
            IOException {
        getObjectMapperInstance().writerWithDefaultPrettyPrinter().writeValue(w, value);
    }

    /*
     * @see ObjectWriter#writeValue(File, Object);
     */
    public static void writeValue(File src, Object value) throws JsonGenerationException, JsonMappingException,
            IOException {
        getObjectMapperInstance().writerWithDefaultPrettyPrinter().writeValue(src, value);
    }

    /**
     * @see ObjectMapper#writeValueAsString(Object)
     * @param value
     * @return
     * @throws JsonProcessingException
     */
    public static String writeValueAsString(Object value) throws JsonProcessingException{
        return getObjectMapperInstance().writeValueAsString(value);
    }

}

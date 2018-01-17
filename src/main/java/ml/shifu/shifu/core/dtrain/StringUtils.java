/*
 * Copyright [2013-2017] PayPal Software Foundation
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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Some String utility functions. Why such class? No any dependency on other jars.
 * 
 * @author pengzhang
 */
public class StringUtils {

    /**
     * Manual split function to avoid depending on guava.
     * 
     * <p>
     * Some examples: "^"=&gt;[, ]; ""=&gt;[]; "a"=&gt;[a]; "abc"=&gt;[abc]; "a^"=&gt;[a, ]; "^b"=&gt;[, b];
     * "^^b"=&gt;[, , b]
     * 
     * @param str
     *            the string to be split
     * @param delimiter
     *            the delimiter
     * @return split string array
     */
    public static String[] split(String str, String delimiter) {
        if(str == null || str.length() == 0) {
            return new String[] { "" };
        }

        List<String> categories = new ArrayList<String>();
        int dLen = delimiter.length();
        int begin = 0;
        for(int i = 0; i < str.length(); i++) {
            if(str.substring(i, Math.min(i + dLen, str.length())).equals(delimiter)) {
                categories.add(str.substring(begin, i));
                begin = i + dLen;
            }
            if(i == str.length() - 1) {
                categories.add(str.substring(begin, str.length()));
            }
        }

        return categories.toArray(new String[0]);
    }

    /**
     * Simple name without name space part.
     * 
     * @param columnName
     *            the column name
     * @return the simple name not including name space part
     */
    public static String getSimpleColumnName(String columnName) {
        String result = columnName;
        // remove name-space in column name to make it be called by simple name
        if(columnName.contains(CommonConstants.NAMESPACE_DELIMITER)) {
            result = columnName.substring(columnName.lastIndexOf(CommonConstants.NAMESPACE_DELIMITER)
                    + CommonConstants.NAMESPACE_DELIMITER.length(), columnName.length());
        }
        return result;
    }

    public static void writeString(DataOutput out, String str) throws IOException {
        if(str == null) {
            out.writeInt(0);
        } else {
            byte[] bytes = str.getBytes("UTF-8");
            out.writeInt(bytes.length);
            for(byte b: bytes) {
                out.write(b);
            }
        }
    }

    public static String readString(DataInput in) throws IOException {
        int size = in.readInt();
        if(size == 0) {
            return null;
        } else {
            byte[] bytes = new byte[size];
            for(int i = 0; i < size; i++) {
                bytes[i] = in.readByte();
            }
            return new String(bytes, "UTF-8");
        }
    }
}

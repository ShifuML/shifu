/*
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/
package ml.shifu.shifu.udf.norm;

import java.math.BigDecimal;
import java.text.DecimalFormat;

public enum PrecisionType {

    FLOAT7 {

        public Float to(double value) {
            return (Double.isNaN(value) ? Float.NaN : Float.parseFloat(DECIMAL_FORMAT.format(value)));
        }

    },
    FLOAT16 {

        public Float to(double value) {
            if (Double.isNaN(value)) {
                return Float.NaN;
            }

            float float16 = toFloat(fromFloat((float) value));
            if (Float.isNaN(float16)) {
                return Float.NaN;
            }

            BigDecimal bdnum = BigDecimal.valueOf(float16);
            if(float16 < 1f && float16 > -1f) {
                bdnum = bdnum.setScale(4, BigDecimal.ROUND_FLOOR);
            } else {
                bdnum = bdnum.setScale(3, BigDecimal.ROUND_FLOOR);
            }
            return bdnum.floatValue();
        }

    },
    FLOAT32 {

        public Float to(double value) {
            return (float) value;
        }

    },
    DOUBLE64 {

        public Double to(double value) {
            return value;
        }

    };

    public static DecimalFormat DECIMAL_FORMAT = new DecimalFormat("#.######");

    public static PrecisionType of(String precisionType) {
        for(PrecisionType pt: PrecisionType.values()) {
            if(pt.toString().equalsIgnoreCase(precisionType)) {
                return pt;
            }
        }
        return FLOAT32;
    }

    // FIXME return as Object need extra cost, see how to reduce the cost
    public abstract Object to(double value);

    // returns all higher 16 bits as 0 for all results
    public static int fromFloat(float fval) {
        int fbits = Float.floatToIntBits(fval);
        int sign = fbits >>> 16 & 0x8000; // sign only
        int val = (fbits & 0x7fffffff) + 0x1000; // rounded value

        if(val >= 0x47800000) // might be or become NaN/Inf
        { // avoid Inf due to rounding
            if((fbits & 0x7fffffff) >= 0x47800000) { // is or must become NaN/Inf
                if(val < 0x7f800000) // was value but too large
                    return sign | 0x7c00; // make it +/-Inf
                return sign | 0x7c00 | // remains +/-Inf or NaN
                        (fbits & 0x007fffff) >>> 13; // keep NaN (and Inf) bits
            }
            return sign | 0x7bff; // unrounded not quite Inf
        }
        if(val >= 0x38800000) // remains normalized value
            return sign | val - 0x38000000 >>> 13; // exp - 127 + 15
        if(val < 0x33000000) // too small for subnormal
            return sign; // becomes +/-0
        val = (fbits & 0x7fffffff) >>> 23; // tmp exp for subnormal calc
        return sign | ((fbits & 0x7fffff | 0x800000) // add subnormal bit
                + (0x800000 >>> val - 102) // round depending on cut off
        >>> 126 - val); // div by 2^(1-(exp-127+15)) and >> 13 | exp=0
    }

    // ignores the higher 16 bits
    public static float toFloat(int hbits) {
        int mant = hbits & 0x03ff; // 10 bits mantissa
        int exp = hbits & 0x7c00; // 5 bits exponent
        if(exp == 0x7c00) {// NaN/Inf
            exp = 0x3fc00; // -> NaN/Inf
        } else if(exp != 0) { // normalized value
            exp += 0x1c000; // exp - 15 + 127
            if(mant == 0 && exp > 0x1c400) // smooth transition
                return Float.intBitsToFloat((hbits & 0x8000) << 16 | exp << 13 | 0x3ff);
        } else if(mant != 0) { // && exp==0 -> subnormal
            exp = 0x1c400; // make it normal
            do {
                mant <<= 1; // mantissa * 2
                exp -= 0x400; // decrease exp by 1
            } while((mant & 0x400) == 0); // while not normal
            mant &= 0x3ff; // discard subnormal bit
        } // else +/-0 -> +/-0
        return Float.intBitsToFloat( // combine all parts
                (hbits & 0x8000) << 16 // sign << ( 31 - 15 )
                        | (exp | mant) << 13); // value << ( 23 - 10 )
    }

    public static void main(String[] args) {
        PrecisionType pt = PrecisionType.DOUBLE64;
        double test = Double.NaN;
        pt = PrecisionType.FLOAT16;
        System.out.println(pt + ":" + pt.to(test));
        double dd = 111.2345679899881234d;
        pt = PrecisionType.DOUBLE64;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT32;
        System.out.println(pt + ":" + pt.to(dd));
        System.out.println("float:" + (float) (dd));
        pt = PrecisionType.FLOAT16;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT7;
        System.out.println(pt + ":" + pt.to(dd));
        float fff = toFloat(fromFloat((float) dd));
        System.out.println("To From:" + fff);
        BigDecimal bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(3, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 3:" + bdnum.floatValue());
        bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(4, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 4:" + bdnum.floatValue());
        System.out.println("");
        dd = 11.2345679899881234d;
        pt = PrecisionType.DOUBLE64;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT32;
        System.out.println(pt + ":" + pt.to(dd));
        System.out.println("float:" + (float) (dd));
        pt = PrecisionType.FLOAT16;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT7;
        System.out.println(pt + ":" + pt.to(dd));
        fff = toFloat(fromFloat((float) dd));
        System.out.println("To From:" + fff);
        bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(3, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 3:" + bdnum.floatValue());
        bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(4, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 4:" + bdnum.floatValue());
        System.out.println("");

        dd = 1.2345679899881234d;
        pt = PrecisionType.DOUBLE64;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT32;
        System.out.println(pt + ":" + pt.to(dd));
        System.out.println("float:" + (float) (dd));
        pt = PrecisionType.FLOAT16;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT7;
        System.out.println(pt + ":" + pt.to(dd));
        fff = toFloat(fromFloat((float) dd));
        System.out.println("To From:" + fff);
        bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(3, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 3:" + bdnum.floatValue());
        bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(4, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 4:" + bdnum.floatValue());
        System.out.println("");

        dd = 0.2345679899881234d;
        pt = PrecisionType.DOUBLE64;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT32;
        System.out.println(pt + ":" + pt.to(dd));
        System.out.println("float:" + (float) (dd));
        pt = PrecisionType.FLOAT16;
        System.out.println(pt + ":" + pt.to(dd));
        pt = PrecisionType.FLOAT7;
        System.out.println(pt + ":" + pt.to(dd));
        fff = toFloat(fromFloat((float) dd));
        System.out.println("To From:" + fff);
        bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(3, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 3:" + bdnum.floatValue());
        bdnum = BigDecimal.valueOf(fff);
        bdnum = bdnum.setScale(4, BigDecimal.ROUND_FLOOR);
        System.out.println("Round 4:" + bdnum.floatValue());
    }
}
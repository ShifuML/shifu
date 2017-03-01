/*
 * Copyright [2013-2015] PayPal Software Foundation
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

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.lang.StringUtils;

/**
 * A helper class for java reflection.
 */
public final class ClassUtils {

    private ClassUtils() {
    }

    /**
     * Only support constructors with no parameters.
     */
    public static final Class<?>[] EMPTY_CLASS_ARRAY = new Class[] {};

    /**
     * This map is used for cache <class, constructor> pairs
     */
    private static final Map<Class<?>, Constructor<?>> CONSTRUCTOR_CACHE = new ConcurrentHashMap<Class<?>, Constructor<?>>();

    /**
     * This map is used for cache <className+'#'+fieldName, Method> pairs
     */
    private static final Map<String, Field> FIELD_CACHE = new ConcurrentHashMap<String, Field>();

    /**
     * This map is used for cache <className+'#'+methodName, Method> pairs. Only first method will be cached here, make
     * sure no same method name for such cache.
     */
    private static final Map<String, Method> METHOD_CACHE = new ConcurrentHashMap<String, Method>();

    /*
     * Given a class instance, return all fields, including fields in super classes.
     */
    @SuppressWarnings("unchecked")
    public static List<Field> getAllFields(Class<?> clazz) {
        if(clazz == null || clazz.equals(Object.class)) {
            return Collections.EMPTY_LIST;
        }

        List<Field> result = new ArrayList<Field>();
        for(Field field: clazz.getDeclaredFields()) {
            result.add(field);
        }
        Class<?> tmpClazz = clazz.getSuperclass();
        while(!Object.class.equals(tmpClazz)) {
            result.addAll(getAllFields(tmpClazz));
            tmpClazz = tmpClazz.getSuperclass();
        }

        return result;
    }

    /*
     * Given a class instance, return all methods, including methods in super classes.
     */
    @SuppressWarnings("unchecked")
    public static List<Method> getAllMethods(Class<?> clazz) {
        if(clazz == null) {
            return Collections.EMPTY_LIST;
        }

        if(clazz.equals(Object.class)) {
            return Collections.EMPTY_LIST;
        }

        List<Method> result = new ArrayList<Method>();
        for(Method method: clazz.getDeclaredMethods()) {
            result.add(method);
        }
        Class<?> tmpClazz = clazz.getSuperclass();
        while(!Object.class.equals(tmpClazz)) {
            result.addAll(getAllMethods(tmpClazz));
            tmpClazz = tmpClazz.getSuperclass();
        }

        return result;
    }

    /*
     * Create an object for the given class. The class should have constructor without any parameters.
     * 
     * @param clazz
     *            class of which an object is created
     * @return a new object
     * @throws RuntimeException
     *             In case any exception for reflection.
     */
    public static <T> T newInstance(Class<T> clazz) {
        T result;
        try {
            @SuppressWarnings("unchecked")
            Constructor<T> meth = (Constructor<T>) CONSTRUCTOR_CACHE.get(clazz);
            if(meth == null) {
                meth = clazz.getDeclaredConstructor(EMPTY_CLASS_ARRAY);
                meth.setAccessible(true);
                CONSTRUCTOR_CACHE.put(clazz, meth);
            }
            result = meth.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return result;
    }

    /*
     * Create an object for the given class. The class should have constructor without any parameters.
     * 
     * @param clazz
     *            class of which an object is created
     * @param parameterClasses
     *            the parameter classes used for constructor
     * @param parameters
     *            the parameters used for new instance
     * @return a new object
     * @throws RuntimeException
     *             In case any exception for reflection.
     */
    public static <T> T newInstance(Class<T> clazz, Class<?>[] parameterClasses, Object[] parameters) {
        T result;
        try {
            @SuppressWarnings("unchecked")
            Constructor<T> meth = (Constructor<T>) CONSTRUCTOR_CACHE.get(clazz);
            if(meth == null) {
                meth = clazz.getDeclaredConstructor(parameterClasses);
                meth.setAccessible(true);
                CONSTRUCTOR_CACHE.put(clazz, meth);
            }
            result = meth.newInstance(parameters);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return result;
    }

    /*
     * Get declared field given field name including fields in super classes. If no field get, return null.
     */
    public static Field getDeclaredFieldIncludeSuper(String fieldName, Class<?> clazz) {
        String key = clazz.getName() + "#" + fieldName;
        Field cacheField = FIELD_CACHE.get(key);
        if(cacheField != null) {
            return cacheField;
        }
        for(Field field: ClassUtils.getAllFields(clazz)) {
            if(field.getName().equals(fieldName)) {
                return field;
            }
        }
        return null;
    }

    /*
     * Return declared method with empty parameter.
     */
    public static Method getDeclaredMethod(String methodName, Class<?> clazz) throws NoSuchMethodException {
        String key = clazz.getName() + "#" + methodName;
        Method cacheMethod = METHOD_CACHE.get(key);
        if(cacheMethod != null) {
            return cacheMethod;
        }
        return clazz.getDeclaredMethod(methodName, ClassUtils.EMPTY_CLASS_ARRAY);
    }

    /*
     * Given a field, return a getter method. Get method should follow common java programming standard.
     */
    public static Method getDeclaredGetter(String name, Class<?> clazz) throws NoSuchMethodException {
        return getDeclaredMethod("get" + StringUtils.capitalize(name), clazz);
    }

    /*
     * Given a field, return a getter method. Get method should follow common java programming standard. Any exception
     * null will be returned.
     */
    public static Method getDeclaredGetterWithNull(String name, Class<?> clazz) {
        try {
            return getDeclaredMethod("get" + StringUtils.capitalize(name), clazz);
        } catch (Exception e) {
            return null;
        }
    }

    /*
     * Given a field, return a setter method. Any exception null will be returned.
     */
    public static Method getDeclaredSetter(String name, Class<?> clazz) throws NoSuchMethodException {
        Method method = getFirstMethodWithName("set" + StringUtils.capitalize(name), clazz);
        if(method == null) {
            throw new NoSuchMethodException();
        }
        return method;
    }

    /*
     * Given a field, return a setter method. Any exception null will be returned.
     */
    public static Method getDeclaredSetterWithNull(String name, Class<?> clazz) {
        return getFirstMethodWithName("set" + StringUtils.capitalize(name), clazz);
    }

    /*
     * Iterate all methods including methods in super class, return the first one; if no method with such name, return
     * null.
     */
    public static Method getFirstMethodWithName(String name, Class<?> clazz) {
        String key = clazz.getName() + "#" + name;
        Method cacheMethod = METHOD_CACHE.get(key);
        if(cacheMethod != null) {
            return cacheMethod;
        }
        List<Method> allMethods = ClassUtils.getAllMethods(clazz);
        Method method = null;
        for(Method f: allMethods) {
            if(f.getName().equals(name)) {
                method = f;
                break;
            }
        }
        return method;
    }

    private static void setFieldValue(Field field, Object instance, Object value) throws IllegalArgumentException,
            IllegalAccessException {
        field.setAccessible(true);
        field.set(instance, value);
    }

    private static Object getFieldValue(Field field, Object instance) throws IllegalArgumentException,
            IllegalAccessException {
        field.setAccessible(true);
        return field.get(instance);
    }

    /*
     * Set filed value according to field name, class instance, object instance and field value. All exceptions are
     * wrapped as RuntimeException.
     */
    public static void setFieldValue(String fieldName, Class<?> clazz, Object instance, Object value) {
        try {
            Field field = getDeclaredFieldIncludeSuper(fieldName, clazz);
            setFieldValue(field, instance, value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /*
     * Return field value according to field name, class, instance.All exceptions are wrapped as RuntimeException.
     */
    public static Object getFieldValue(String fieldName, Class<?> clazz, Object instance) {
        try {
            Field field = getDeclaredFieldIncludeSuper(fieldName, clazz);
            return getFieldValue(field, instance);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /*
     * Call method through java reflection.
     */
    public static Object invokeMethod(Method method, Object instance, Object... parameters) {
        try {
            method.setAccessible(true);
            return method.invoke(instance, parameters);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /*
     * Call method through java reflection. TODO getFirstMethodWithName should be changed by finding method with method
     * name and parameter types.
     */
    public static Object invokeMethod(String name, Class<?> clazz, Object instance, Object... parameters) {
        try {
            Method method = getFirstMethodWithName(name, clazz);
            method.setAccessible(true);
            return method.invoke(instance, parameters);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

}

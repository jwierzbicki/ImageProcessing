using System;
using System.Runtime.InteropServices;

namespace GUI
{
    public static class NativeMethods
    {
        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        public extern static void RemoveColorFromImage(IntPtr imageData, int length, int imageHeight, int imageWidth, int bytesPerPixel, int color);

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        public extern static void FilterImage(IntPtr inputImage, IntPtr outputImage, IntPtr mask, int length, int imageHeight, int imageWidth, int bytesPerPixel);

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        public extern static void ConvertImageToGrayscale(IntPtr inputImage, IntPtr outputGrayscale, int length, int imageHeight, int imageWidth, int bytesPerPixel);

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        public extern static void ThresholdImage(IntPtr inputImage, IntPtr outputGrayscale, int length, int imageHeight, int imageWidth, int bytesPerPixel, int parts = 2);

        [DllImport("kernel32.dll")]
        public static extern void RtlZeroMemory(IntPtr dst, UIntPtr length);

        [DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);
    }
}

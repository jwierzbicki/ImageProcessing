using Microsoft.Win32.SafeHandles;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media.Imaging;

namespace GUI
{
    public static class Converters
    {
        public static Pixel[] ParseImageToPixelArray(byte[] image, int bytesPerPixel)
        {
            var pixelArray = new List<Pixel>();

            for (int i = 0; i < image.Length; i += bytesPerPixel)
            {
                pixelArray.Add(new Pixel { Red = image[i], Green = image[i + 1], Blue = image[i + 2] });
            }

            return pixelArray.ToArray();
        }

        public static Pixel[][] ParseImageToPixelMatrix(byte[] image, int height, int width, int bytesPerPixel)
        {
            var pixelMatrix = new List<List<Pixel>>();

            for (int row = 0; row < height; row++)
            {
                var pixelArray = new List<Pixel>();

                for (int i = row * width * bytesPerPixel; i < width * (row + 1) * bytesPerPixel; i += bytesPerPixel)
                {
                    pixelArray.Add(new Pixel { Red = image[i], Green = image[i + 1], Blue = image[i + 2] });
                }

                pixelMatrix.Add(pixelArray);
            }

            var temp = pixelMatrix.Select(x => x.ToArray()).ToArray();

            return temp;
        }

        public static byte[] ParsePixelArrayToByteArray(Pixel[] pixels, int bytesPerPixel)
        {
            var bytes = new List<byte>();

            for (int i = 0; i < pixels.Length; i++)
            {
                bytes.Add(pixels[i].Red);
                bytes.Add(pixels[i].Green);
                bytes.Add(pixels[i].Blue);
            }

            return bytes.ToArray();
        }

        public static byte[] ParsePixelMatrixToByteArray(Pixel[][] pixels, int height, int width, int bytexPerPixel)
        {
            var bytes = new List<byte>();

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    bytes.Add(pixels[i][j].Red);
                    bytes.Add(pixels[i][j].Green);
                    bytes.Add(pixels[i][j].Blue);
                }
            }

            return bytes.ToArray();
        }

        public static void ClearPixels(byte[] imageBytes, int imageWidth, int imageHeight, int bytesPerPixel)
        {
            for (int row = 0; row < imageHeight; row++)
            {
                for (int col = 0; col < imageWidth; col++)
                {
                    for (int k = 0; k < bytesPerPixel; k++)
                    {
                        if (k == 2)
                        {
                            imageBytes[row * (imageWidth * bytesPerPixel) + col * bytesPerPixel + k] = 0;
                        }
                    }
                }
            }
        }

        public static Bitmap ByteArrayToBitmap(byte[] bytes)
        {
            using (var memoryStream = new MemoryStream(bytes))
            {
                memoryStream.Position = 0;
                return new Bitmap(memoryStream);
            }
        }

        public static BitmapSource BitmapToBitmapSource(Bitmap bitmap)
        {
            IntPtr hBitmap = bitmap.GetHbitmap();

            try
            {
                return Imaging.CreateBitmapSourceFromHBitmap(hBitmap, IntPtr.Zero, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());
            }
            finally
            {
                NativeMethods.DeleteObject(hBitmap);
            }
        }

        public static byte[] BitmapToByteArray(Bitmap bmp)
        {
            using (var memoryStream = new MemoryStream())
            {
                bmp.Save(memoryStream, ImageFormat.Bmp);
                return memoryStream.ToArray();
            }
        }

        public static Bitmap ConvertTo24bpp(Bitmap orig)
        {
            var bmp24 = new Bitmap(orig.Width, orig.Height, PixelFormat.Format24bppRgb);
            using (var gr = Graphics.FromImage(bmp24))
            {
                gr.DrawImage(orig, new Rectangle(0, 0, bmp24.Width, bmp24.Height));
            }

            return bmp24;
        }

        public static void CopyBytesIntoBitmap(MyImage img)
        {
            BitmapData bmData = img.Bitmap.LockBits(new Rectangle(0, 0, img.Bitmap.Width, img.Bitmap.Height), ImageLockMode.ReadWrite, img.Bitmap.PixelFormat);
            IntPtr pNative = bmData.Scan0;
            Marshal.Copy(img.Bytes, 0, pNative, img.Bytes.Length);
            img.Bitmap.UnlockBits(bmData);
        }
    }
}

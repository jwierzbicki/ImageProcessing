using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace GUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public List<MyImage> Images { get; set; }

        public MainWindow()
        {
            InitializeComponent();

            Initialize();
        }

        private void Initialize()
        {
            Images = new List<MyImage>();

            // Create first image
            var image = new MyImage();
            image.Bitmap = new Bitmap("hamster24.bmp");
            image.Bytes = Converters.BitmapToByteArray(image.Bitmap);
            Images.Add(image);

            MainImage.Source = Converters.BitmapToBitmapSource(image.Bitmap);
        }

        private void RemoveColorFromImage(Color color)
        {
            // Create new image
            var newImage = (MyImage)Images.Last().Clone();

            // Allocate memory via Marshal
            IntPtr tempArray = Marshal.AllocHGlobal(newImage.DataLength);
            Marshal.Copy(newImage.Bytes, newImage.DataOffset, tempArray, newImage.DataLength);

            // Do C/C++ operation
            RemoveColorFromImage(tempArray, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel, (int)color);

            // Copy back
            Marshal.Copy(tempArray, newImage.Bytes, newImage.DataOffset, newImage.DataLength);
            // Free memory
            Marshal.FreeHGlobal(tempArray);

            newImage.Bitmap = Converters.ByteArrayToBitmap(newImage.Bytes);

            Images.Add(newImage);

            // Set image source
            MainImage.Source = Converters.BitmapToBitmapSource(newImage.Bitmap);
        }

        private void BtnRemoveRed_Click(object sender, RoutedEventArgs e)
        {
            RemoveColorFromImage(Color.Red);
        }

        private void BtnRemoveGreen_Click(object sender, RoutedEventArgs e)
        {
            RemoveColorFromImage(Color.Green);
        }

        private void BtnRemoveBlue_Click(object sender, RoutedEventArgs e)
        {
            RemoveColorFromImage(Color.Blue);
        }

        private void BtnBlur_Click(object sender, RoutedEventArgs e)
        {
            // Create new image
            var newImage = (MyImage)Images.Last().Clone();

            // Allocate memory via Marshal
            IntPtr inputArray = Marshal.AllocHGlobal(newImage.DataLength);
            IntPtr outputArray = Marshal.AllocHGlobal(newImage.DataLength);
            // Copy input data
            Marshal.Copy(newImage.Bytes, newImage.DataOffset, inputArray, newImage.DataLength);
            // Clear output array
            RtlZeroMemory(outputArray, new UIntPtr((uint)newImage.DataLength));

            // Do C/C++ operation
            BlurImage(inputArray, outputArray, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel);

            // Copy back
            Marshal.Copy(outputArray, newImage.Bytes, newImage.DataOffset, newImage.DataLength);
            // Free memory
            Marshal.FreeHGlobal(inputArray);
            Marshal.FreeHGlobal(outputArray);

            newImage.Bitmap = Converters.ByteArrayToBitmap(newImage.Bytes);

            Images.Add(newImage);

            // Set image source
            MainImage.Source = Converters.BitmapToBitmapSource(newImage.Bitmap);
        }

        private void BtnUndo_Click(object sender, RoutedEventArgs e)
        {
            if(Images.Count > 1)
            {
                // Remove last one
                Images.RemoveAt(Images.Count - 1);

                // Set one before last as current image
                MainImage.Source = Converters.BitmapToBitmapSource(Images.Last().Bitmap);
            }
        }

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static void RemoveColorFromImage(IntPtr imageData, int length, int imageHeight, int imageWidth, int bytesPerPixel, int color);

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static void BlurImage(IntPtr inputImage, IntPtr outputImage, int length, int imageHeight, int imageWidth, int bytesPerPixel);

        [DllImport("kernel32.dll")]
        static extern void RtlZeroMemory(IntPtr dst, UIntPtr length);
    }
}

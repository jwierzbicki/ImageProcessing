using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace GUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public List<MyImage> Images { get; set; }

        public int UndoLeft { get; set; }

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
            UndoLeftLabel.Content = ++UndoLeft;
        }

        private void BtnRemoveGreen_Click(object sender, RoutedEventArgs e)
        {
            RemoveColorFromImage(Color.Green);
            UndoLeftLabel.Content = ++UndoLeft;
        }

        private void BtnRemoveBlue_Click(object sender, RoutedEventArgs e)
        {
            RemoveColorFromImage(Color.Blue);
            UndoLeftLabel.Content = ++UndoLeft;
        }

        private void BtnBlur_Click(object sender, RoutedEventArgs e)
        {
            float[] kernel = {
                1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
                1 / 9.0f, 1 / 9.0f, 1 / 9.0f,
                1 / 9.0f, 1 / 9.0f, 1 / 9.0f
            };

            // Create new image
            var newImage = (MyImage)Images.Last().Clone();

            // Allocate memory via Marshal
            IntPtr inputArray = Marshal.AllocHGlobal(newImage.DataLength);
            IntPtr outputArray = Marshal.AllocHGlobal(newImage.DataLength);
            IntPtr mask = Marshal.AllocHGlobal(kernel.Length * sizeof(float));
            // Copy input data
            Marshal.Copy(newImage.Bytes, newImage.DataOffset, inputArray, newImage.DataLength);
            Marshal.Copy(kernel, 0, mask, kernel.Length);
            // Clear output array
            RtlZeroMemory(outputArray, new UIntPtr((uint)newImage.DataLength));

            // Do C/C++ operation
            FilterImage(inputArray, outputArray, mask, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel);

            // Copy back
            Marshal.Copy(outputArray, newImage.Bytes, newImage.DataOffset, newImage.DataLength);
            // Free memory
            Marshal.FreeHGlobal(inputArray);
            Marshal.FreeHGlobal(outputArray);
            Marshal.FreeHGlobal(mask);

            newImage.Bitmap = Converters.ByteArrayToBitmap(newImage.Bytes);

            Images.Add(newImage);

            // Set image source
            MainImage.Source = Converters.BitmapToBitmapSource(newImage.Bitmap);
            UndoLeftLabel.Content = ++UndoLeft;
        }

        private void BtnGrayscale_Click(object sender, RoutedEventArgs e)
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
            ConvertImageToGrayscale(inputArray, outputArray, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel);

            // Copy back
            Marshal.Copy(outputArray, newImage.Bytes, newImage.DataOffset, newImage.DataLength);
            // Free memory
            Marshal.FreeHGlobal(inputArray);
            Marshal.FreeHGlobal(outputArray);

            newImage.Bitmap = Converters.ByteArrayToBitmap(newImage.Bytes);

            Images.Add(newImage);

            // Set image source
            MainImage.Source = Converters.BitmapToBitmapSource(newImage.Bitmap);
            UndoLeftLabel.Content = ++UndoLeft;
        }

        private void BtnThreshold_Click(object sender, RoutedEventArgs e)
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
            int parts = int.Parse(((ComboBoxItem)ThresholdComboBox.SelectedItem).Content.ToString());
            ThresholdImage(inputArray, outputArray, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel, parts);

            // Copy back
            Marshal.Copy(outputArray, newImage.Bytes, newImage.DataOffset, newImage.DataLength);
            // Free memory
            Marshal.FreeHGlobal(inputArray);
            Marshal.FreeHGlobal(outputArray);

            newImage.Bitmap = Converters.ByteArrayToBitmap(newImage.Bytes);

            Images.Add(newImage);

            // Set image source
            MainImage.Source = Converters.BitmapToBitmapSource(newImage.Bitmap);
            UndoLeftLabel.Content = ++UndoLeft;
        }

        private void BtnUndo_Click(object sender, RoutedEventArgs e)
        {
            if(Images.Count > 1)
            {
                // Remove last one
                Images.RemoveAt(Images.Count - 1);

                // Set one before last as current image
                MainImage.Source = Converters.BitmapToBitmapSource(Images.Last().Bitmap);

                UndoLeftLabel.Content = --UndoLeft;
            }
        }

        private void OpenFileButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dlg = new OpenFileDialog();
            dlg.Filter = "Image Files (*.jpg;*.jpeg;*.png;*.bmp;*.gif)|*.jpg;*.jpeg;*.png;*.bmp;*.gif|All Files (*.*)|*.*";
            bool? result = dlg.ShowDialog();
            if(result == false)
            {
                return;
            }

            string fileName = dlg.FileName;

            var image = new MyImage();
            image.Bitmap = Converters.ConvertTo24bpp(new Bitmap(System.Drawing.Image.FromFile(fileName)));
            image.Bytes = Converters.BitmapToByteArray(image.Bitmap);
            Images.Clear();
            Images.Add(image);
            MainImage.Source = Converters.BitmapToBitmapSource(image.Bitmap);
        }

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static void RemoveColorFromImage(IntPtr imageData, int length, int imageHeight, int imageWidth, int bytesPerPixel, int color);

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static void FilterImage(IntPtr inputImage, IntPtr outputImage, IntPtr mask, int length, int imageHeight, int imageWidth, int bytesPerPixel);

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static void ConvertImageToGrayscale(IntPtr inputImage, IntPtr outputGrayscale, int length, int imageHeight, int imageWidth, int bytesPerPixel);

        [DllImport(@"../../../../x64/Debug/ImageProcessing.dll", CallingConvention = CallingConvention.Cdecl)]
        private extern static void ThresholdImage(IntPtr inputImage, IntPtr outputGrayscale, int length, int imageHeight, int imageWidth, int bytesPerPixel, int parts = 2);

        [DllImport("kernel32.dll")]
        static extern void RtlZeroMemory(IntPtr dst, UIntPtr length);
    }
}

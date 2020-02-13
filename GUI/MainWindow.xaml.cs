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

        public Dictionary<string, float[]> FilterMasks { get; set; }

        public MainWindow()
        {
            InitializeComponent();

            Initialize();
        }

        private void Initialize()
        {
            Images = new List<MyImage>();

            FilterMasks = InitializeMasks();

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
            NativeMethods.RemoveColorFromImage(tempArray, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel, (int)color);

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

        private void BtnFilter_Click(object sender, RoutedEventArgs e)
        {
            string kernelName = ((ComboBoxItem)FilterComboBox.SelectedItem).Content.ToString();
            var kernel = FilterMasks[kernelName];

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
            NativeMethods.RtlZeroMemory(outputArray, new UIntPtr((uint)newImage.DataLength));

            // Do C/C++ operation
            NativeMethods.FilterImage(inputArray, outputArray, mask, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel);

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
            NativeMethods.RtlZeroMemory(outputArray, new UIntPtr((uint)newImage.DataLength));

            // Do C/C++ operation
            NativeMethods.ConvertImageToGrayscale(inputArray, outputArray, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel);

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
            NativeMethods.RtlZeroMemory(outputArray, new UIntPtr((uint)newImage.DataLength));

            // Do C/C++ operation
            int parts = int.Parse(((ComboBoxItem)ThresholdComboBox.SelectedItem).Content.ToString());
            NativeMethods.ThresholdImage(inputArray, outputArray, newImage.DataLength, newImage.Height, newImage.Width, newImage.BytesPerPixel, parts);

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
            Bitmap bmp = new Bitmap(System.Drawing.Image.FromFile(fileName));
            image.Bitmap = Converters.ConvertTo24bpp(bmp);
            image.Bytes = Converters.BitmapToByteArray(image.Bitmap);
            Images.Clear();
            Images.Add(image);
            MainImage.Source = Converters.BitmapToBitmapSource(image.Bitmap);
        }

        private static Dictionary<string, float[]> InitializeMasks()
        {
            var dict = new Dictionary<string, float[]>();

            dict["EdgeDetection1"] = new float[]
            {
                1, 0, -1,
                0, 0, 0,
                -1, 0, 1
            };

            dict["EdgeDetection2"] = new float[]
            {
                0, 1, 0,
                1, -4, 1,
                0, 1, 0
            };

            dict["EdgeDetection3"] = new float[]
            {
                -1, -1, -1,
                -1, 8, -1,
                -1, -1, -1
            };

            dict["Sharpen"] = new float[]
            {
                0, -1, 0,
                -1, 5, -1,
                0, -1, 0
            };

            dict["BoxBlur"] = new float[] {
                1/9.0f, 1/9.0f, 1/9.0f,
                1/9.0f, 1/9.0f, 1/9.0f,
                1/9.0f, 1/9.0f, 1/9.0f
            };

            dict["GaussianBlur"] = new float[]
            {
                1/16.0f, 1/8.0f, 1/16.0f,
                1/8.0f, 1/4.0f, 1/8.0f,
                1/16.0f, 1/8.0f, 1/16.0f
            };

            return dict;
        }
    }
}

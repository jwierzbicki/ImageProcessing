using System;
using System.Drawing;

namespace GUI
{
    public class MyImage : ICloneable
    {
        public int Width { get => Bitmap.Width; }

        public int Height { get => Bitmap.Height; }

        public byte[] Bytes { get; set; }

        public int DataOffset { get => BitConverter.ToInt32(Bytes, 10); }

        public int DataLength { get => Bytes.Length - DataOffset;  }

        public int BytesPerPixel {
            get
            {
                switch (this.Bitmap.PixelFormat)
                {
                    case System.Drawing.Imaging.PixelFormat.Format24bppRgb:
                        return 3;
                    case System.Drawing.Imaging.PixelFormat.Format32bppRgb:
                    case System.Drawing.Imaging.PixelFormat.Format32bppArgb:
                        return 4;
                    default:
                        // unknown
                        return 0;
                }
            }
        }

        public Bitmap Bitmap { get; set; }

        public object Clone()
        {
            var bitmapClone = new Bitmap(this.Width, this.Height, this.Bitmap.PixelFormat);
            using (Graphics gr = Graphics.FromImage(bitmapClone))
            {
                gr.DrawImage(this.Bitmap, new Rectangle(0, 0, bitmapClone.Width, bitmapClone.Height));
            }

            var image = new MyImage();
            image.Bytes = Converters.BitmapToByteArray(bitmapClone);
            image.Bitmap = bitmapClone;
            return image;
        }
    }
}

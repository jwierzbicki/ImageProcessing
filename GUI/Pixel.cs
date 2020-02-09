using System.Runtime.InteropServices;

namespace GUI
{
    [StructLayout(LayoutKind.Sequential)]
    public class Pixel
    {
        public byte Red { get; set; }

        public byte Green { get; set; }

        public byte Blue { get; set; }
    }
}
